def evaluate_step(model, loader, metrics, device):
    """
    Perform one evaluation step for the combined model.

    Args:
        model (nn.Module): The combined model (GNN + Drug Model + ResNet).
        loader (DataLoader): DataLoader for the evaluation dataset.
        metrics (MetricTracker): Metric tracker for evaluation.
        device (torch.device): The device to use (CPU or GPU).

    Returns:
        dict: Computed metrics as key-value pairs.
    """
    metrics.increment()
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for batch in loader:
            cell_graph, drug_vector, targets = batch['cell_graph'], batch['drug_vector'], batch['targets']
            cell_graph = cell_graph.to(device)
            drug_vector = drug_vector.to(device)
            targets = targets.to(device)

            # Forward pass through the model
            outputs = model(cell_graph, drug_vector)

            # Update metrics with predictions and targets
            metrics.update(
                outputs.squeeze(),
                targets.squeeze(),
                cell_lines=batch['cell_lines'].to(device).squeeze(),
                drugs=batch['drugs'].to(device).squeeze(),
            )

    # Return computed metrics
    return {key: value.item() for key, value in metrics.compute().items()}

def train_step(model, optimizer, loader, config, device):
    """
    Perform one training step for the combined model (GNN + Drug Model + ResNet).
    """
    loss_fn = nn.MSELoss()
    total_loss = 0
    model.train()  # Set the model to training mode

    for batch in loader:
        cell_graph, drug_vector, targets = batch['cell_graph'], batch['drug_vector'], batch['targets']
        cell_graph = cell_graph.to(device)
        drug_vector = drug_vector.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass through the combined model
        outputs = model(cell_graph, drug_vector)

        # Compute loss
        loss = loss_fn(outputs.squeeze(), targets.squeeze())

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
        optimizer.step()

        # Collect loss
        total_loss += loss.item()

    return total_loss / len(loader)  # Return average loss for the epoc

def train_model(config, train_dataset, validation_dataset=None, callback_epoch=None):
    """
    Train the combined model with training and optional validation datasets.
    """
    # Define data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["optimizer"]["batch_size"],
        drop_last=True,
        shuffle=True
    )
    if validation_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=config["optimizer"]["batch_size"],
            drop_last=False,
            shuffle=False
        )

    # Initialize the model
    gnn_model = ModularGNN(**config["gnn"])  # Initialize GNN with config
    drug_mlp = DrugMLP(input_dim=config["drug"]["input_dim"], embed_dim=config["gnn"]["output_dim"])
    resnet = ResNet(embed_dim=config["gnn"]["output_dim"], hidden_dim=config["resnet"]["hidden_dim"])
    combined_model = CombinedModel(gnn=gnn_model, drug_mlp=drug_mlp, resnet=resnet)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(combined_model.parameters(), config["optimizer"]["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    early_stop = EarlyStop(config["optimizer"]["stopping_patience"])

    # Device setup
    device = torch.device(config["env"]["device"])
    combined_model.to(device)

    # Define metrics
    metrics = torchmetrics.MetricTracker(torchmetrics.MetricCollection({
        "R_cellwise_residuals": GroupwiseMetric(
            metric=torchmetrics.functional.pearson_corrcoef,
            grouping="drugs",
            average="macro",
            residualize=True
        ),
        "R_cellwise": GroupwiseMetric(
            metric=torchmetrics.functional.pearson_corrcoef,
            grouping="cell_lines",
            average="macro",
            residualize=False
        ),
        "MSE": torchmetrics.MeanSquaredError()
    }))
    metrics.to(device)

    # Training loop
    best_val_target = None
    for epoch in range(config["env"]["max_epochs"]):
        # Train for one epoch
        train_loss = train_step(combined_model, optimizer, train_loader, config, device)

        # Update learning rate scheduler
        lr_scheduler.step(train_loss)

        # Validation
        if validation_dataset is not None:
            validation_metrics = evaluate_step(combined_model, val_loader, metrics, device)
            if epoch > 0 and config["optimizer"]["use_momentum"]:
                best_val_target = 0.2 * best_val_target + 0.8 * validation_metrics['R_cellwise_residuals']
            else:
                best_val_target = validation_metrics['R_cellwise_residuals']
        else:
            best_val_target = None

        # Log progress
        if callback_epoch is None:
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation R_cellwise_residuals: {best_val_target}")
        else:
            callback_epoch(epoch, best_val_target)

        # Early stopping
        if early_stop(train_loss):
            print(f"Stopping early at epoch {epoch + 1}.")
            break

    return best_val_target, combined_model
import torch

def train(model, train_loader, criterion, optimizer, epoch, num_epochs, device):
    # Set model in a train mode
    train_loss = 0.0
    correct_predictions = 0

    for inputs, targets in train_loader:
        # Set your inputs and target to the current device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        # Forward pass
        # 1. Calculate model's outputs
        outputs = model(inputs)
        # 2. Calculate loss
        loss = criterion(outputs, targets)
        # 3. Zero gradients
        optimizer.zero_grad()
        # 4. Add loss value to the overall train loss
        train_loss += loss.item()
        # Backward pass and optimization
        # 1. Do backward pass
        loss.backward()
        # 2. Do optimizer step
        optimizer.step()

    # Calculate average loss
    avg_loss = train_loss / len(train_loader)

    print(f'Epoch [{epoch + 1:03}/{num_epochs:03}] | Train Loss: {avg_loss:.4f}')
    return train_loss/len(train_loader)

# Defining a Validation Function
def validate(model, val_loader, criterion, device):
    # Set model in a evaluation mode
    model.eval()
    val_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():# Use torch method to avoid calculating/storing gradients
        for inputs, targets in val_loader:
            # Set your inputs and target to the current device
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            # Forward pass
            # 1. Calculate model's outputs
            outputs = model(inputs)
            # 2. Calculate loss
            loss = criterion(outputs, targets)
            # 3. Add loss value to the overall validation loss
            val_loss += loss.item()

    # Calculate average loss
    avg_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f} ')
    return avg_loss

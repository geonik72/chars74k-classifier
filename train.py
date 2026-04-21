import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import CharacterClassification

if __name__ == '__main__': #necessary for Windows to avoid multiprocessing issues with DataLoader

    train_split = 0.8 # Use 80% of the data for training and 20% for testing
    batch_size = 256
    num_epochs = 10
    lr = 0.0005
    weight_decay = 1e-4
    counter = 0

    # Training Transforms (with Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.RandomRotation(15), # Rotates image randomly between -15 and +15 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Shifts image horizontally/vertically up to 10%
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(), # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)) # Normalize pixel values to [-1, 1]
    ])

    # Testing Transforms (NO Augmentation, just resizing and normalizing)
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_dir = 'data\\English\\Fnt'

    

    # Load the base datasets with their specific transforms
    full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    full_test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)

    #  Create a single list of indices and shuffle it 
    indices = torch.randperm(len(full_train_dataset)).tolist()
    train_size = int(train_split * len(full_train_dataset))

    # Split the datasets using those exact indices so they match perfectly
    train_dataset = Subset(full_train_dataset, indices[:train_size]) # Use the first (train_split)% of the shuffled indices for training
    test_dataset = Subset(full_test_dataset, indices[train_size:]) # Use the remaining (1-train_split)% of the shuffled indices for testing

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = CharacterClassification() # Instantiate the model

    # Set up device, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss() # Handles softmax internally, so we can output raw logits from the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Adam optimizer with weight decay for regularization

    # Training loop
    for epoch in range(num_epochs):

        print(f'Starting epoch {epoch+1}/{num_epochs}...')
        model.train() # Set model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            counter += 1
            if counter % 20 == 0:
                print(f'Number of batches processed: {counter}/{len(train_loader)}')

            images, labels = images.to(device), labels.to(device) # Move data to device
            optimizer.zero_grad() # Clear gradients from previous step
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update model parameters
            running_loss += loss.item() * images.size(0) # Accumulate loss for the entire epoch

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluate on test set
        if (epoch + 1) % 2 == 0:  # evaluate every 2 epochs
            model.eval()
            correct, total = 0, 0
            with torch.no_grad(): # Disable gradient calculation for evaluation
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f'Accuracy on test set: {accuracy:.4f}')

    torch.save(model.state_dict(), 'model.pth') # Save the trained model's state for later use
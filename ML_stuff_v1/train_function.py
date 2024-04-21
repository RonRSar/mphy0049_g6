import torch
from tqdm.auto import tqdm
from typing import Dict, List
from sklearn.metrics import precision_score, f1_score, recall_score
from pathlib import Path

### Training
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    
    # Put model in train mode
    model.train() 

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0 
           
    # Loop through the dataloader 
    for batch, (X, y) in enumerate(dataloader):
            
        # Send to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X) 

        # 2. Caclulate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

        # # logging the accuracy and loss after 10 batches
        # if batch % 10 == 0:
        #     batch_acc = (y_pred_class == y).sum().item()/len(y_pred)
        #     wandb.log({"Batch Accuracy of 10": batch_acc})
        #     wandb.log({"Batch Loss of 10": loss.item()})
    
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer:torch.optim.Optimizer,
              device: torch.device):

    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    y_pred = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            y_pred = torch.cat((y_pred, test_pred_labels), dim=0)
            y_true = torch.cat((y_true, y), dim=0)
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = ((y_pred == y_true).sum().item()/len(y_pred))
    y_true, y_pred = y_true.to('cpu'), y_pred.to('cpu')

    precision_test = precision_score(y_true, y_pred, average='macro')   
    recall_test = recall_score(y_true, y_pred, average='macro')
    f1_test = f1_score(y_true, y_pred, average='macro')
    

    return test_loss, test_acc, precision_test, recall_test, f1_test, y_pred, y_true

# 1. Create a train function that takes in various model parameters + optimizer + dataloaders + loss function
def train_data(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          MODEL_SAVE_PATH: str,
          device: torch.device,
          best_acc = 0) -> Dict[str, List]:

  # 2. Create empty results dictionary
  results = {"Train Loss": [],
             "Train Acc": [],
             "Test Loss": [],
             "Test Acc": [],
             "Test Precision": [],
             "Test Recall": [],
             "Test F1": []}
  
  # 3. Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        print("train_step")
        
        test_loss, test_acc, precision_test, f1_test, recall_test, y_pred, y_true = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
        print('test_step')
        # 4. print out what's happenin
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        # 5. Update results dictionary
        results["Train Loss"].append(train_loss)
        results["Train Acc"].append(train_acc)
        results["Test Loss"].append(test_loss)
        results["Test Acc"].append(test_acc)
        results['Test Precision'].append(precision_test)
        results['Test Recall'].append(recall_test)
        results['Test F1'].append(f1_test)
        
        # # log metrics to wandb
        # wandb.log({'epoch': epoch,
        #            "Train Loss": train_loss})
        # wandb.log({'epoch': epoch,
        #            "Test Loss": test_loss})
        # wandb.log({'epoch': epoch,
        #            "NEC Epoch Train Acc": train_acc})
        # wandb.log({'epoch': epoch,
        #            "Test Acc": test_acc})
        # wandb.log({'epoch': epoch,
        #            "Test Precision": precision_test})
        # wandb.log({'epoch': epoch,
        #            "Test Recall": recall_test})
        # wandb.log({'epoch': epoch,
        #            "Test F1": f1_test})
        
        # If the accuracy is better than the previous best, save the model

        if test_acc > best_acc:
          if abs(test_acc-train_acc) <=0.3:
            
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
              
            print(f"Model checkpoint saved for new best accuracy: {best_acc}")
            print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
            print('y_true')
            print(y_true)
            print('y_pred')
            print(y_pred)

            # wandb.log({'epoch': epoch,
            #           "Max Epoch Test Accuracy": best_acc})
        
    # 6. Return the filled results at the end of the epochs
  return results
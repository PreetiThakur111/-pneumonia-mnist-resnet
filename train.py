
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):  
  model.to(device) 
  for epoch in range(epochs):       
    print(f"Epoch {epoch+1}/{epochs}")        
    # Training       
model.train()        
running_loss = 0.0       
running_corrects = 0        
total = 0         
for inputs, labels in train_loader:             
  inputs = inputs.to(device)          
  labels = labels.to(device)             
  optimizer.zero_grad()            
  outputs = model(inputs)            
  loss = criterion(outputs, labels)            
  loss.backward()             
  optimizer.step()              
  running_loss += loss.item() * inputs.size(0)             _, preds = torch.max(outputs, 1)      
  running_corrects += torch.sum(preds == labels.data)            
  total += labels.size(0)       
  train_loss = running_loss / total      
  train_acc = running_corrects.double() / total          
  print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")         
  # Validation        
model.eval()         
val_loss = 0.0         
val_corrects = 0         
val_total = 0         
with torch.no_grad():             
for inputs, labels in val_loader:               
  inputs = inputs.to(device)                 
  
  labels = labels.to(device)                 
  outputs = model(inputs)                 
  loss = criterion(outputs, labels)                 
  val_loss += loss.item() * inputs.size(0)               _, preds = torch.max(outputs, 1)            
  val_corrects += torch.sum(preds == labels.data)            
  val_total += labels.size(0)         
  val_loss = val_loss / val_total         
  val_acc = val_corrects.double() / val_total        
  print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}
")
  


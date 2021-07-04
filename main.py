import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn

from torchsummary.torchsummary import summary

import torchvision
import torchvision.transforms as transforms

import my_model
import timm.loss as loss

import os

def validation(model, criterion, val_data):
  model.eval()
  test_loss = 0
  correct = 0
  total = 0

  with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(val_data):
          inputs, targets = inputs.to(device), targets.to(device)

          with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
          
          test_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0)
          correct += predicted.eq(targets).sum().item()
          
  accuracy = 100. * correct / total

  return accuracy


if __name__ == "__main__":
  tfm = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2023, 0.1994, 0.2010])
  ])

  val_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010])
  ])

  train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10/',
                                                      train=True,
                                                      download=True,
                                                      transform=tfm
                                                    )

  val_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10/',
                                                    train=False,
                                                    download=True,
                                                    transform=val_tfm)

  batch_size = 256
  train_data = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

  val_data = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    cudnn.benchmark = True
  else:
    device = torch.device("cpu")

  model = my_model.myCNN(in_channels=64, num_classes=len(classes), layers_per_stage=[2, 4, 6, 2], norm_type='group').to(device)
  summary(model, input_size=(3, 32, 32))

  learning_rate = 0.1
  epochs = 200

  criterion = loss.LabelSmoothingCrossEntropy(smoothing=0.1)

  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)  
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 170], gamma=0.1)

  scaler = amp.GradScaler(enabled=True)

  best_acc = 0.0

  save_weights = "myCNN_2_last.pth"
  save_best_weights = "myCNN_2_best.pth"

  save_path = "./weights"
  os.makedirs(save_path, exist_ok=True)

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    total = 0
    correct = 0

    for i, data in enumerate(train_data):
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      with amp.autocast(enabled=True):
        output = model(inputs)

      loss = criterion(output, labels)
      scaler.scale(loss).backward()

      scaler.step(optimizer=optimizer)
      scaler.update()

      running_loss += loss.item()
      _, predicted = output.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()

      if i % 50 == 0:
        print('[Epochs: %d/%d][Iter: %d/%d] loss: %.3f, acc: %.3f%%, lr: %f' % 
              (epoch, epochs, i, len(train_data), running_loss / 100, 
               100.*correct/total, optimizer.param_groups[0]['lr']))
        running_loss = 0.0

    scheduler.step()
    acc = validation(model, criterion, val_data)

    print('[Epochs: %d/%d] Validation accuracy: %.3f%%\n' % (epoch, epochs, acc))
    torch.save(model.state_dict(), 
              os.path.join(save_path, save_weights))

    if acc > best_acc:
      print("\n**************************************************")
      print("Best weight saved! [%.2f%% ==> %.2f%% improved.]" % (best_acc, acc))
      print("**************************************************\n\n")
      best_acc = acc

      torch.save(model.state_dict(), 
                os.path.join(save_path, save_best_weights))

  print('\nTraining ends...')



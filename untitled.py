
from hyperdash import Experiment

exp = Experiment("test",api_key_getter = get_hyperdash_api)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda")


def train(model,dl,optimizer,criterion):
    mdoel.train()
    n_correct, n_total = 0, 0
    for i, data in enumerate(dl, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        n_correct += ((torch.max(F.softmax(answer, dim=1), 1)[1].view(label.size())== label).sum().item())
        n_total += batch_size

        exp.metric('loss',loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    acc = 100.0 * n_correct / n_total
    exp.metric('train acc',acc)
    exp.metric('running loss',running_loss)
    print('Loss: {}'.format(running_loss)


def validate(model,dl,criterion):
    n_correct, n_total = 0, 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dl, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # forward 
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            n_correct += ((torch.max(F.softmax(answer, dim=1), 1)[1].view(label.size())== label).sum().item())
            n_total += batch_size

            exp.metric('loss',loss.item())
            loss.backward()
            running_loss += loss.item()


        acc = 100.0 * n_correct / n_total
        exp.metric('val acc',acc)
        exp.metric('running loss',running_loss)
        print('Loss: {}'.format(running_loss)


# loop over the dataset multiple times
for epoch in range(5):
    train(model,trainloader,optimizer,criterion)
    validate(model,trainloader,criterion)

exp.end()
print('Finished Training')
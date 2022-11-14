import torch
import numpy as np
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model,dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)

        # Compute prediction error
        pred,_ = model(X)
        # print(type(pred))
        # print(pred.shape,y.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/len(X)
    return loss.item(), 100*correct
    # return loss, 100*correct




def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.type(torch.FloatTensor).to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct
    # return 100*correct


# def train_mul(dataloader, models, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     for model in models:
#         model.train()

#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
#         optimizer.zero_grad()
#         train_pre=[]
#         vote_correct_train=0
#         models_correct_train=[0 for i in range(len(models))]
#         avg_loss=0
#         for i,model in enumerate(models):
#         # Compute prediction error
            
#             pred = model(X[:,i,:].unsqueeze(dim=1))
#             loss = loss_fn(pred, y)
#             avg_loss+=loss.item()
#             # Backpropagation
#             loss.backward()
#             _,prediction=torch.max(pred,1) #按行取最大值
#             pre_num=prediction.cpu().numpy()
#             models_correct_train[i]+=(pre_num==y.cpu().numpy()).sum()
#             train_pre.append(pre_num)
#         avg_loss/=8
#         optimizer.step()
#         arr=np.array(train_pre)
#         train_pre.clear()
#         result=[Counter(arr[:,i]).most_common(1)[0][0] for i in range(len(X))]
#         vote_correct_train+=(result == y.cpu().numpy()).sum()
        
#     correct = vote_correct_train/len(X)
#     return avg_loss, 100*correct
#     # return loss, 100*correct

# def test_mul(dataloader, models, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     for model in models:
#         model.eval()
#     test_loss, correct = 0, 0
#     test_pre=[]
#     models_correct_train=[0 for i in range(len(models))]
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.type(torch.FloatTensor).to(device), y.to(device)
#             for i,model in enumerate(models):
#                 pred = model(X[:,i,:].unsqueeze(dim=1))
#                 _,prediction=torch.max(pred,1) #按行取最大值
#                 pre_num=prediction.cpu().numpy()
#                 models_correct_train[i]+=(pre_num==y.cpu().numpy()).sum()
#                 test_pre.append(pre_num)
#             arr=np.array(test_pre)
#             test_pre.clear()
#             result=[Counter(arr[:,i]).most_common(1)[0][0] for i in range(len(X))]
#             correct+=(result == y.cpu().numpy()).sum()
#             # test_loss += loss_fn(torch.tensor(result), y).item()
#     # test_loss /= num_batches
#     correct /= size
#     # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#     # return test_loss, 100*correct
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% \n")
#     return 100*correct


def train_mul(dataloader, models, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for model in models:
        model.train()
    for batch,(X,y) in enumerate(dataloader):
        X, y = X.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        train_pre=[]
        vote_correct_train=0
        models_correct_train=[0 for i in range(len(models))]
        avg_loss=0
        for i,model in enumerate(models):
        # Compute prediction error
            pred = model(X[:,i,:].unsqueeze(dim=1))
            loss = loss_fn(pred, y)
            avg_loss+=loss.item()
            # Backpropagation
            loss.backward()
            _,prediction=torch.max(pred,1) #按行取最大值
            pre_num=prediction.cpu().numpy()
            models_correct_train[i]+=(pre_num==y.cpu().numpy()).sum()
            train_pre.append(pre_num)
        avg_loss/=len(models)
        optimizer.step()
        arr=np.array(train_pre)
        train_pre.clear()
        result=[Counter(arr[:,i]).most_common(1)[0][0] for i in range(len(X))]
        vote_correct_train+=(result == y.cpu().numpy()).sum()
        
    correct = vote_correct_train/len(X)
    return avg_loss, 100*correct
    # return loss, 100*correct

def test_mul(dataloader, models, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    for model in models:
        model.eval()
    test_loss, correct = 0, 0
    test_pre=[]
    result_list=[]
    models_correct_train=[0 for i in range(len(models))]
    features=[[]for i in range(len(models))]
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.type(torch.FloatTensor).to(device), y.to(device)
            for i,model in enumerate(models):
                print(X[:,i,:].unsqueeze(dim=1).shape)
                pred = model(X[:,i,:].unsqueeze(dim=1))

                # print(model.outputfeature.cpu().numpy().shape)
                # features[i].append(model.outputfeature.cpu().numpy())
                _,prediction=torch.max(pred,1) #按行取最大值
                pre_num=prediction.cpu().numpy()
                models_correct_train[i]+=(pre_num==y.cpu().numpy()).sum()
                test_pre.append(pre_num)
                
            arr=np.array(test_pre)
            test_pre.clear()
            result=[Counter(arr[:,i]).most_common(1)[0][0] for i in range(len(X))]
            correct+=(result == y.cpu().numpy()).sum()
            # test_loss += loss_fn(torch.tensor(result), y).item()
        # for i in range(len(models)):
        #     features[i]=np.concatenate((features[i]),axis=0)
    # test_loss /= num_batches
    # features=np.array(features)
    # print(features.shape)
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # return test_loss, 100*correct
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% \n")
    return 100*correct #,features

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# 2*EEG+EOG
def train_model11(dataloader, model, loss_fn, optimizer,seqlen):
    size = len(dataloader.dataset)*seqlen
    num_batches = len(dataloader)

    model.train()
    # print(size)
    # print(len(dataloader))
    trainloss,correct=0,0

    for batch, (X_0,X_1,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        # print('X0.shape',X0.shape)
        X1 = X_1.type(torch.FloatTensor).to(device)
        key_padding_mask = torch.ones(X0.shape[0], X0.shape[1]).to(device)  # [batch,seq]
        
        y = y.reshape(-1)
        # print('y.shape',y.shape)
        # Compute prediction error
        pred = model(X0,X1,key_padding_mask)
        # print("output shape:",pred.shape)
        pred = pred.reshape(-1,5)
        # print('pred.shape',pred.shape)
        # pred /= torch.sum(pred,dim=-1).unsqueeze(dim = -1)
        # pred = torch.clamp(pred,0,1)
        # pred = model(X0)
        # print(pred.shape)#,pred)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        trainloss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    trainloss = trainloss/num_batches
    correct = correct/size
    return trainloss, 100*correct
    # return loss, 100*correct

def test_model11(dataloader, model, loss_fn, seqlen):
    size = len(dataloader.dataset)*seqlen
    # print("size:",size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,X_1,y in dataloader:
            print(X_0.shape,y.shape)
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            y = y.reshape(-1)
            key_padding_mask = torch.ones(X0.shape[0], X0.shape[1]).to(device)  # [batch,seq]
            pred = model(X0,X1,key_padding_mask)

            # pred = model(X0)
            pred = pred.reshape(-1,5)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= size
        avg_test_loss = test_loss/num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return  avg_test_loss,100*correct

def train_model11_ch4(dataloader, model, loss_fn, optimizer,seqlen):
    size = len(dataloader.dataset)*seqlen
    num_batches = len(dataloader)

    model.train()
    # print(size)
    # print(len(dataloader))
    trainloss,correct=0,0

    for batch, (X_0,X_1,X_2,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        # print('X0.shape',X0.shape)
        X1 = X_1.type(torch.FloatTensor).to(device)
        X2 = X_2.type(torch.FloatTensor).to(device)
        key_padding_mask = torch.ones(X0.shape[0], X0.shape[1]).to(device)  # [batch,seq]
        
        y = y.reshape(-1)
        # print('y.shape',y.shape)
        # Compute prediction error
        pred = model(X0,X1,X2,key_padding_mask)
        # print("output shape:",pred.shape)
        pred = pred.reshape(-1,5)
        # print('pred.shape',pred.shape)
        # pred /= torch.sum(pred,dim=-1).unsqueeze(dim = -1)
        # pred = torch.clamp(pred,0,1)
        # pred = model(X0)
        # print(pred.shape)#,pred)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        trainloss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    trainloss = trainloss/num_batches
    correct = correct/size
    return trainloss, 100*correct
    # return loss, 100*correct

def test_model11_ch4(dataloader, model, loss_fn, seqlen):
    size = len(dataloader.dataset)*seqlen
    # print("size:",size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,X_1,X_2,y in dataloader:
            print(X_0.shape,y.shape)
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            X2 = X_2.type(torch.FloatTensor).to(device)
            y = y.reshape(-1)
            key_padding_mask = torch.ones(X0.shape[0], X0.shape[1]).to(device)  # [batch,seq]
            pred = model(X0,X1,X2,key_padding_mask)

            # pred = model(X0)
            pred = pred.reshape(-1,5)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= size
        avg_test_loss = test_loss/num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return  avg_test_loss,100*correct


def train_3ch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print(size)
    model.train()
    # print(enumerate(dataloader).shape)
    for batch, (X_0,X_1,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        X1 = X_1.type(torch.FloatTensor).to(device)
        # X2 = X_2.type(torch.FloatTensor).to(device)
        # X3 = X_3.type(torch.FloatTensor).to(device)
        # Compute prediction error
        pred = model(X0,X1)
        # print('pred.shape',pred.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/size
    return loss.item(), 100*correct
    # return loss, 100*correct

def test_3ch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,X_1,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            pred = model(X0,X1)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return  test_loss,100*correct


# 2*EEG+EOG+EMG
def train_4ch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print("train size:",size)
    model.train()
    # print(enumerate(dataloader).shape)
    for batch, (X_0,X_1,X_2,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        X1 = X_1.type(torch.FloatTensor).to(device)
        X2 = X_2.type(torch.FloatTensor).to(device)
        # Compute prediction error
        pred = model(X0,X1,X2)
        # print('y.shape',y.shape)
        loss = loss_fn(pred, y)
        # print('pred.shape',pred.shape)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/len(X_0)
    return loss.item(), 100*correct
    # return loss, 100*correct

def test_4ch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    # print("test size:",size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,X_1,X_2,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            X2 = X_2.type(torch.FloatTensor).to(device)
            pred = model(X0,X1,X2)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return  test_loss,100*correct

def train_salient(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print("Train size:",size)
    model.train()
    # print(len(dataloader))
    for batch, (X_0,X_1,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        # print('X0.shape',X0.shape)
        X1 = X_1.type(torch.FloatTensor).to(device)
        
        y = y.reshape(-1)
        # print('y.shape',y.shape)
        # Compute prediction error
        pred = model(X0,X1)
        
        pred = pred.reshape(-1,5)
        # print('pred.shape',pred.shape)
        # pred /= torch.sum(pred,dim=-1).unsqueeze(dim = -1)
        # pred = torch.clamp(pred,0,1)
        # pred = model(X0)
        # print(pred.shape)#,pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/(len(X_0)*20)
    return loss.item(), 100*correct
    # return loss, 100*correct

def test_salient(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    print("size:",size)
    num_batches = len(dataloader)
    print("num_batches:",num_batches)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,X_1,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            y = y.reshape(-1)
            pred = model(X0,X1)
            # pred = model(X0)
            pred = pred.reshape(-1,5)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches*20
    correct /= size*20
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # return  100*correct
    return test_loss, 100*correct

def train_1ch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print(size)
    model.train()
    # print(enumerate(dataloader).shape)
    for batch, (X_0,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print('X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape:',X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        # X1 = X_1.type(torch.FloatTensor).to(device)
        # X2 = X_2.type(torch.FloatTensor).to(device)
        # X3 = X_3.type(torch.FloatTensor).to(device)
        # Compute prediction error

        pred = model(X0)
        # print('pred.shape',pred.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/size
    return loss.item(), 100*correct
    # return loss, 100*correct

def test_1ch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            # X1 = X_1.type(torch.FloatTensor).to(device)
            pred = model(X0)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return  test_loss,100*correct

def train_printnet(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # print(size)
    model.train()
    # print(enumerate(dataloader).shape)
    for batch, (X_0,X_1,X_2,X_3,y) in enumerate(dataloader):
        # print(list(enumerate(dataloader)))
        # print(X_0.type(torch.FloatTensor).to(device).shape,y.type(torch.LongTensor).shape)
        X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
        X1 = X_1.type(torch.FloatTensor).to(device)
        X2 = X_2.type(torch.FloatTensor).to(device)
        X3 = X_3.type(torch.FloatTensor).to(device)
        # Compute prediction error
        pred = model(X0,X1,X2,X3)
        # print('pred.shape',pred.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_0)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()/len(X_0)
    return loss.item(), 100*correct
    # return loss, 100*correct





def test_printnet(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X_0,X_1,X_2,X_3,y in dataloader:
            X0, y = X_0.type(torch.FloatTensor).to(device), y.type(torch.LongTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            X2 = X_2.type(torch.FloatTensor).to(device)
            X3 = X_3.type(torch.FloatTensor).to(device)
            pred = model(X0,X1,X2,X3)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from datetime import datetime
from modular import engine
from extra_functions import set_seed, extract_metrics_hier
from torchvision import models


# Base from https://arxiv.org/abs/1709.09890


########################## BCNN with VGG16 Architecture ########################

# class VGGBlock(nn.Module):
#     """
#     Convolutional block used in VGG-style architecture. 
    
#     It consists of a sequence of covolutional layers, each followed by a RELU 
#     activation and batch normalization. After specified number of convolutional 
#     layers, a max pooling is applied. 
    
#     Args:
#         in_channels (int): Number of input channels
#         out_channels (int): Number of output channels
#         num_cov (int): Number of covolutional layers in the block. 
    
#     """
    
#     def __init__(self, in_channels:int, out_channels:int, num_conv: int):
#         super(VGGBlock, self).__init__()
#         layers = []
#         for i in range(num_conv):
#             layers +=[
#                 nn.Conv2d(
#                     in_channels = in_channels if i == 0 else out_channels,
#                     out_channels = out_channels,
#                     kernel_size = 3,
#                     padding = 1
#                 ),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(out_channels)
#             ]
#         layers +=[nn.MaxPool2d(kernel_size=2,stride=2)]  
#         self.block = nn.Sequential(*layers)  
            
#     def forward(self, x):
#         return self.block(x)    
        
# class CoarseBlock(nn.Module):
#     """
#     Fully connected classification block for coarse level predictions. 
    
#     It consists of a flattens block that extracted features maps and passes them
#     through a fully connected layer wit ReLu activations, batch normalization and 
#     dropout regularization. 
    
#     Args: 
#         feature_extractor (nn.module): A convolutional feature extractor whose outputs 
#         defines de input size for the first fully connected layer.
#         coarse_classes (int): Number of outputs classes for the coarse level-prediction.
#         input_shape (tuple,optional): Shape of the input tensor (channels, height, width). 
#         Default is (3,64,64).    
    
#     """
    
#     def __init__(self,feature_extractor,coarse_classes:int, input_shape=(3,64,64)):        
#         super(CoarseBlock,self).__init__()   
        
#         # adapt flattened feature by resolution
#         with torch.no_grad():
#             dummy = torch.zeros(1, *input_shape)
#             feat = feature_extractor(dummy)
#             in_features = feat.numel()
        
        
#         self.flatten = nn.Flatten()
#         self.fc_layers = nn.Sequential(
#             nn.Linear(in_features, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.5),
                        
#             nn.Linear(256,256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.5),
#         )
        
#         self.out = nn.Linear(256,coarse_classes)
    
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.fc_layers(x)
#         x = self.out(x) # logits
#         return x
     
         
# class BcnnVGG(nn.Module):
#     """
#     BCNN VGG-Style fir multi-label classification
    
#     Args:
#         input_shape (int): Number of input channels
#         dim_outputs (list): List with the number of output classes for each hierarchy level. 
#         resolution (int, optional): Input image resolution. Default is 64
#         levels (int): Number of hierarchy levels (2 or 3). Default is 2  
         
#     """
    
    
#     def __init__(self,input_shape:int, dim_outputs:list,            
#                  resolution: int = 64,levels=2):
#         super(BcnnVGG,self).__init__()
        
#         self.levels =levels
        
#         if self.levels not in [2,3]:
#             raise ValueError('Error: Level must be 2 or 3.')
        
#         if len(dim_outputs)!=levels:
#             raise ValueError('Error: dim_outputs should be of length levels.')
                       
#         self.block_1 = VGGBlock(in_channels=input_shape, out_channels=64,num_conv=2)
#         self.block_2 = VGGBlock(in_channels=64, out_channels=128,num_conv=2)
#         feature_extractor = nn.Sequential(self.block_1, self.block_2)
#         self.coarse1 = CoarseBlock(
#             feature_extractor,
#             input_shape = (3,resolution,resolution),
#             coarse_classes = dim_outputs[0]
#         )
#         self.block_3 = VGGBlock(in_channels=128, out_channels=256,num_conv=3)
#         feature_extractor = nn.Sequential(self.block_1, self.block_2,self.block_3)
#         if self.levels==3:
#             self.coarse2 = CoarseBlock(
#                 feature_extractor,
#                 input_shape=(3,resolution,resolution),
#                 coarse_classes=dim_outputs[1]        
#             )
#         self.block4 = VGGBlock(in_channels=256,out_channels=512,num_conv=3)
#         self.block5 = VGGBlock(in_channels=512,out_channels=512,num_conv=3)
#         feature_extractor = nn.Sequential(
#             self.block_1, self.block_2,self.block_3,
#             self.block4,self.block5
#         )
#         self.fine = CoarseBlock(
#             feature_extractor,
#             coarse_classes=dim_outputs[levels-1]
#         )
        
    
#     def forward(self,x):
#         # Block 1 and 2
#         x = self.block_1(x)
#         x = self.block_2(x)
#         c1 = self.coarse1(x)  
        
#         # Coarse level 1
#         c1_pred = F.softmax(c1,dim=1)
#         x = self.block_3(x)
        
#         # Coarse level 2 (only if hierarchical level == 3)
#         if self.levels ==3:
#             c2 = self.coarse2(x)
#             c2_pred = F.softmax(c2,dim=1)
            
#         # Final pred
#         x = self.block4(x)
#         x = self.block5(x)
#         fine = self.fine(x)  
#         fine_pred = F.softmax(fine,dim=1)

#         # Return predictions depending on the hierarchy depth
#         if self.levels ==3:
#             return c1_pred, c2_pred, fine_pred
#         elif self.levels ==2: 
#             return c1_pred, fine_pred       

class BcnnVGG(nn.Module):
    def __init__(self, dim_outputs:list , levels: int =2, weights_directory = None):
        super().__init__()
        
        self.levels = levels
        self.weights_path = weights_directory + '/vgg16-397923af.pth'
        
        if self.levels not in [2,3]:
            raise ValueError('Error: Level must be 2 or 3.')
        
        if len(dim_outputs)!=levels:
            raise ValueError('Error: dim_outputs should be of length levels.')

        base = models.vgg16(weights= None)
        state_dict = torch.load(self.weights_path, map_location = 'cpu')
        base.load_state_dict(state_dict, strict=False)
        
        
        # Split pretrained VGG into two parts
        self.features_block1 = base.features[:19]  
        self.features_block2 = base.features[19:24]
        self.features_block3 = base.features[24:]   
        
        # Coarse level 1 
        self.coarse_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, dim_outputs[0])
        )

        # Coarse level 2 
        if self.levels ==3:
            self.coarse_2 = nn.Sequential(
                nn.AdaptiveAvgPool2d((7,7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, dim_outputs[1])
            )
        
        # Fine level
        self.fine_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, dim_outputs[self.levels-1])
        )
        
        # Initialize with pretrained weights for shared parts
        self._init_from_pretrained(base)
    
    def _init_from_pretrained(self, base):
        # Copy matching weights
        pretrained_dict = base.state_dict()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
    
    def forward(self, x):
        # Block 1
        x = self.features_block1(x)
        c1_pred = self.coarse_1(x)
        x = self.features_block2(x)
        if self.levels ==3:
            c2_pred = self.coarse_2(x)
        fine_pred = self.fine_head(x)
                
        # Return predictions depending on the hierarchy depth
        if self.levels ==3:
            return c1_pred, c2_pred, fine_pred
        elif self.levels ==2: 
            return c1_pred, fine_pred     
        


############################ BCNN Model definition #############################


def hierarchical_loss(outputs, targets,criterion,levels,alphas):
   
    if len(alphas)!=levels:
        raise ValueError('Error: The length of alphas must be equal to the number of levels')

    loss  = 0
    for l in range(levels):
        loss += alphas[l] * criterion(outputs[l],targets[l])
   
    return loss
    
def get_alpha_values(epoch, alphas, thresholds = None):
    
    """
        Set the value of hyperparamter alpha based on the specified thresholds an current epoch.

        Args:
            epoch (int): Current epoch number
            alphas (list): List of alpha values for each level. If alpha changes across epochs, provide
                           a list of lists instead (one list per epoch range)
            thresholds (list, optional): If alpha remains constant, set it to None.
                                         Otherwise, provide a list of epoch thresholds indicating when
                                         alpha should change 
        Returns:
            list: Values of alpha corresponding to the current epoch
                        
        Example:
            Case 1 - No changes : 
                        alpha = [0.5,0.5]
                        threshold = None
                        
            Case 2 - Changes at epoch 20 and 50 : 
                        alpha = [[0.5,0.5],[0.3,0.7],[0.1,0.9]]
                        threshold = [20,50]                   
        
                        
        """

    
    if thresholds is None:
        alpha=alphas
    else: 
        n_cuts = len(thresholds)
        if ((n_cuts+1)!=len(alphas)):
            raise ValueError('Error: The length of alphas must be equal to the number of thresholds +1 ')

        alpha = alphas[0]    
        for i, value in enumerate(thresholds):
            if epoch >= value:
                alpha = alphas[i+1]
        
    return alpha

class BCNN_Model:

    def __init__(
            self, weights_directory,dim_outputs, 
            model_name: str = 'vgg16',device: torch.device = None, 
            seed: int = 666,levels:int  = 2
        ):
        
        # Main class initializations
        self.model_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.weights_directory = weights_directory
        self.dim_outputs = dim_outputs
        self.model_name = model_name
        self.device = device
        self.seed = seed
        self.levels = levels
        
        set_seed(self.seed)

        # Load model and weights
        if self.model_name == 'vgg16':
            self.model = BcnnVGG(
                dim_outputs=self.dim_outputs,
                levels = self.levels,
                weights_directory = self.weights_directory
            )
            #self.weights_path = self.weights_directory + '/.....pth'
        else:
            raise ValueError('Unsupported model. Select one of vgg16 or others.')  
                           
        self.model.to(self.device)
       
    def train(
        self, hyperparameters: dict, train_loader, val_loader, 
        verbose: bool = True
    ):

        """
        Trains the model instance using the specified data loaders and training parameters.
        Not all training parameters are supported. Sample hyperparameters dictionary:
            HYPERPARAMETERS = {
                'loss_fn': {'type': 'CrossEntropyLoss','alphas':[0.5,0.5]}, 
                'optimizer': 'Adam', 
                'lr': 5e-4, 
                'epochs': 40, 
                'scheduler': {'type': 'StepLR', 'step_size': 10, 'gamma': 0.1}, 
                'early_stopping': {'patience': 10, 'delta': 0.005}
            }

        Args:
            hyperparameters (dict): Dictionary containing training parameters such as loss function, 
                                    optimizer type, learning rate, scheduler settings, early stopping, and epochs.
            train_loader (DataLoader): PyTorch DataLoader containing the training dataset.
            val_loader (DataLoader): PyTorch DataLoader containing the validation dataset.
            verbose (bool, optional): Whether to print training progress. Defaults to True.
        """

        set_seed(self.seed)

        self.hyperparameters = hyperparameters

        # Loss function
        loss_fn_spec = hyperparameters['loss_fn']
        loss_fn_alpha = loss_fn_spec['alpha']
        thresholds_alpha = loss_fn_spec['thresholds']
        
        
        if loss_fn_spec['criterion'] == 'CrossEntropyLoss':
            loss_fn = nn.CrossEntropyLoss()
            
        else:
            raise ValueError('Unsupported loss function. Select one of CrossEntropyLoss.')
        
        # Optimizer
        if hyperparameters['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                params = self.model.parameters(), lr = hyperparameters['lr']
            )
        elif hyperparameters['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                params = self.model.parameters(), lr = hyperparameters['lr']
            )
        else:
            raise ValueError('Unsupported optimizer. Select one of Adam or SGD.')
        
        # Scheduler
        scheduler_spec = hyperparameters['scheduler']
        if scheduler_spec['type'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size = scheduler_spec['step_size'], gamma = scheduler_spec['gamma']
            )
        elif scheduler_spec['type'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max = scheduler_spec['T_max']
            )
        elif scheduler_spec['type'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr = scheduler_spec['max_lr'], epochs = hyperparameters['epochs'], steps_per_epoch = len(train_loader)
            )
        else:
            raise ValueError('Unsupported scheduler. Select one of StepLR, CosineAnnealingLR or OneCycleLR.')
        
        # Early stopping
        early_stop_criteria = hyperparameters['early_stopping']
        if early_stop_criteria is not None:
            early_stopping = engine.EarlyStopping(
                patience = early_stop_criteria['patience'],
                delta = early_stop_criteria['delta']
            )
        else:
            early_stopping = None

        
        ###################### Main test-train loop ############################

        epochs = hyperparameters['epochs']
        model = self.model
        device = self.device
        model.to(device)
        
        results = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [[] for _ in range(self.levels)],
            'test_acc': [[] for _ in range(self.levels)]
            }
        
        if verbose:
            print(f'\nStarting training! Model: {self.model_name} (ID: {self.model_id})\n')
        start = time.time()
        
        for epoch in range(epochs):
            
            # Get alphas value
            curr_alpha = get_alpha_values(epoch, loss_fn_alpha, thresholds_alpha)
            print(curr_alpha)

            # Train step            
            model.train()
            train_loss = 0
            train_acc = [0 for _ in range(self.levels)]

            for imgs, targets in train_loader:
                
                imgs = imgs.to(device)
                targets = tuple(t.to(device) for t in targets)

                outputs = model(imgs)
                loss =  hierarchical_loss(
                    outputs,targets,criterion = loss_fn,alphas = curr_alpha,
                    levels = self.levels
                )  
                train_loss += loss 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate and accumulate accuracy metric across all batches
                for l in range(self.levels):
                    y_pred_class = torch.argmax(torch.softmax(outputs[l], dim=1), dim=1)
                    train_acc[l] += (y_pred_class == targets[l]).sum().item()/len(outputs[l])                
                    
            train_acc = [x / len(train_loader) for x in train_acc]
            train_loss = train_loss/len(train_loader)
            
            # Test step
            model.eval()
            with torch.inference_mode():
                test_loss = 0
                test_acc = [0 for _ in range(self.levels)]
                
                for imgs, targets in val_loader:
                    
                    imgs = imgs.to(device)
                    targets = tuple(t.to(device) for t in targets)

                    test_pred = model(imgs)
                    test_loss +=  hierarchical_loss(
                        test_pred,targets,criterion = loss_fn,alphas = curr_alpha,
                        levels = self.levels
                    )  
              
                    # Calculate and accumulate accuracy metric across all batches
                    for l in range(self.levels):
                        y_pred_class = torch.argmax(torch.softmax(test_pred[l], dim=1), dim=1)
                        test_acc[l] += (y_pred_class == targets[l]).sum().item()/len(test_pred[l])                
                        
                test_acc = [x / len(val_loader) for x in test_acc]
                test_loss = test_loss/len(val_loader)     
                
            # Early stopping
            if early_stopping is not None:
                early_stopping(test_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break    
                
            # Adjust learning rate
            if scheduler is not None:
                scheduler.step()     
                        
            # Prints                  
            if verbose:
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.5f} | test_loss: {test_loss:.5f} |"
                    f"train_acc: {', '.join([f'{x:.5f}' for x in train_acc])} | "
                    f"test_acc: {', '.join([f'{x:.5f}' for x in test_acc])}"
                )   
                
            # Update results dictionary
            results['test_loss'].append(test_loss)
            results['train_loss'].append(train_loss)
            
            for l in range(self.levels):
                results['train_acc'][l].append(train_acc[l])
                results['test_acc'][l].append(test_acc[l])     
            
       

        elapsed = time.time() - start
        if verbose:
            print(f'\nTraining Finished! Time Elapsed: {elapsed:.2f} sec.')
        
        # Get metrics
        self.train_results = extract_metrics_hier(results)
    
    
    def predict(self, test_loader):
        
        """
        Generates predictions for the specified samples using the trained model instance.

        Args:
            test_loader (DataLoader): PyTorch DataLoader containing the test dataset.
        """
        
        self.model.eval()
        labels = [[] for _ in range(self.levels)]
        preds = [[] for _ in range(self.levels)]
        probs = [[] for _ in range(self.levels)]
        logits = [[] for _ in range(self.levels)]
        
        with torch.no_grad():
            for img, targets in test_loader:
                
                img = img.to(self.device)
                targets = tuple(t.to(self.device) for t in targets)
                
                output = self.model(img)
                
                for l in range(self.levels):
                    prob = F.softmax(output[l],dim=1)
                    pred = output[l].argmax(dim=1)
                    probs[l].append(prob)
                    preds[l].append(pred)
                    logits[l].append(output[l])
                    labels[l].append(targets[l])
                    
            for l in range(self.levels):
                probs[l] = torch.cat(probs[l])
                preds[l] = torch.cat(preds[l])
                logits[l] = torch.cat(logits[l])
                labels[l] = torch.cat(labels[l])
        
        return labels, probs, preds, logits
                    
                    
                    
                    
        
                    
                
        
        

        
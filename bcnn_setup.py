import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from datetime import datetime
from modular import engine
from extra_functions import set_seed, extract_metrics_hier
from torchvision import models
from itertools import product



# Base from https://arxiv.org/abs/1709.09890


################################ BCNN modules ##################################

# VGG16 architecure

class BcnnVGG(nn.Module):
    
    """
    
    PyTorch model intance implenting a Branch Convolutional Neural Network based on 
    the VGG16 architecture.
    
    Args:
        dim_outputs (list): Number of output nodes for each level of hierarchy.
        levels (int,optional): Number of levels in the hierarchy. Options are 2 or 3.
                               Defaults to 2.
        weights_directory (str): Path to the directory containing pre-trained weigths.    
    
    """
    
    
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
        
        
        # Split pretrained VGG ino 3 parts
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
        x = self.features_block1(x)
        c1_pred = self.coarse_1(x)
        x = self.features_block2(x)
        if self.levels==3:
            c2_pred = self.coarse_2(x)
        fine_pred = self.fine_head(x)
                
        # Return predictions depending on the hierarchy depth
        if self.levels==3:
            return c1_pred, c2_pred, fine_pred
        elif self.levels==2: 
            return c1_pred, fine_pred     
        
# Resnet50 architecure

class BcnnResnet50(nn.Module):
    
    """
    PyTorch model intance implenting a Branch Convolutional Neural Network based on 
    the Resnet50 architecture.
    
    Args:
        dim_outputs (list): Number of output nodes for each level of hierarchy.
        levels (int,optional): Number of levels in the hierarchy. Options are 2 or 3.
                               Defaults to 2.
        weights_directory (str): Path to the directory containing pre-trained weigths.    
    
    """
    def __init__(self, dim_outputs:list , levels: int = 2, weights_directory = None):
        super().__init__()
            
        self.levels = levels
        self.weights_path = weights_directory + '/resnet50-0676ba61.pth'
        base = models.resnet50(weights= None)
        
        state_dict = torch.load(self.weights_path, map_location = 'cpu')
        base.load_state_dict(state_dict, strict=False)

        # Define Blocks
        self.block1 = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2
        ) 
        self.block2 = base.layer3
        self.block3 = base.layer4

        # Coarse level 1 
        self.coarse_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, dim_outputs[0])
        )        
        
        # Coarse level 2 
        if self.levels ==3:
            self.coarse_2 = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(1024, dim_outputs[1])
            )
                
        # Fine level
        self.fine_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, dim_outputs[levels-1])
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
        x = self.block1(x)
        c1_pred = self.coarse_1(x)
        x = self.block2(x)
        if self.levels ==3:
            c2_pred = self.coarse_2(x)
        x = self.block3(x)
        fine_pred = self.fine_head(x)
                
        # Return predictions depending on the hierarchy depth
        if self.levels ==3:
            return c1_pred, c2_pred, fine_pred
        elif self.levels ==2: 
            return c1_pred, fine_pred     
        
# Densenet121 architecure

class BcnnDensenet121(nn.Module):
    
    """
    
    PyTorch model intance implenting a Branch Convolutional Neural Network based on 
    the Densenet121 architecture.
    
    Args:
        dim_outputs (list): Number of output nodes for each level of hierarchy.
        levels (int,optional): Number of levels in the hierarchy. Options are 2, 3 or 4.
                               Defaults to 2.
        weights_directory (str): Path to the directory containing pre-trained weigths.    
    
    """
    def __init__(self, dim_outputs:list , levels: int=2, weights_directory = None):
        super().__init__()
            
        self.levels = levels
        base = models.densenet121(weights = None)
        
        if weights_directory is not None:
            weights_path = weights_directory + '/densenet121-a639ec97.pth' 
            state_dict = torch.load(weights_path, map_location = 'cpu')
            base.load_state_dict(state_dict, strict=False)
        
        if self.levels not in [2,3,4]:
            raise ValueError('Error: Level must be 2, 3 or 4.')
        
        if len(dim_outputs)!=levels:
            raise ValueError('Error: dim_outputs should be of length levels.')
        
        # Split Densenet into four parts
        self.features_block1 = base.features[:5]
        self.features_block2 = base.features[5:8]
        self.features_block3 = base.features[8:10]
        self.features_block4 = base.features[10:]
                
        # Coarse level 1 
        self.coarse_1 =  nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(
                in_features=256,
                out_features=dim_outputs[0], 
                bias=True
            ),           
        )
        
        if levels>=3:
            # Coarse level 2 
            self.coarse_2 =  nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(
                    in_features=256,
                    out_features=dim_outputs[1], 
                    bias=True
                ),           
            )
        
        if levels==4: 
            # Coarse level 3 
            self.coarse_3 =  nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(
                    in_features=512,
                    out_features=dim_outputs[2], 
                    bias=True
                ),           
            )
                
        # Fine level
        self.fine_head =  nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(
                in_features=1024,
                out_features=dim_outputs[(levels-1)], 
                bias=True
            ),           
        )
    def _init_from_pretrained(self, base):
        # Copy matching weights
        pretrained_dict = base.state_dict()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
            
    def forward(self, x):
        x = self.features_block1(x)
        c1_pred = self.coarse_1(x)
        x = self.features_block2(x)
        if self.levels>=3:
            c2_pred = self.coarse_2(x)
        x = self.features_block3(x)
        if self.levels==4:
            c3_pred = self.coarse_3(x)
        x = self.features_block4(x)
        fine_pred = self.fine_head(x)
        
        if self.levels ==2:           
            return c1_pred, fine_pred 
        if self.levels ==3:           
            return c1_pred, c2_pred, fine_pred 
        if self.levels ==4:           
            return c1_pred, c2_pred, c3_pred, fine_pred        
                

############################ BCNN Model definition #############################


def hierarchical_loss(outputs, targets,criterion,levels,alphas):
    
    """
        Loss function with multiple terms.  
        
        Args:
            outputs : output(s) of the PyTorch model
            targets (tuple): Target values for each level
            criterion (torch.nn.Module): The criterion to compute the loss
            levels (int): Number of loss terms .
            alphas (list): Weights for each loss term 
            
        Returns:
            toch.tensor: the combined loss value. 
    
    """
   
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
    
    """
        A wrapper for training and evaluating a Branch Convolutional Neural Network (BCNN).
        
        This class handles:
            - Model initialization with pre-trained weights.
            - Initialization of model instances.
            
        Args:
            weights_directory (str): Path to the directory containing pre-trained weigths.
            dim_outputs (list): Number of output nodes for each level of hierarchy. 
            model_name (str, optional): Model to use ('vgg16' of 'resnet50'). Defaults to 'vgg16'
            device (torch.device, optional): Computation device. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 666.
            levels (int, optional): Number of levels in the hierarchy. Defaults to 2.
            
        Attributes: 
            model_id (str): Unique identifier for the model intance (based on datetime).
            weights_directory (str): Path to the pre-trained weights. 
            dim_outputs (list): Number of output nodes for each level of hierarchy.
            model_name (str): Architecture used by the model.
            device (torch.device): Device on which the model runs.
            seed (int): Random seed used for reproducibility
            model (torch.nn.Module): The PyTorch model instance with the custom classification head.
            hyperparameters (dict or None): Dictionary of training hyperparameters (set later).
            train_results (dict or None): Stores training and evaluation metrics (set later). 
    
    """

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
        elif self.model_name == 'resnet50':
            self.model = BcnnResnet50(
                dim_outputs=self.dim_outputs,
                levels = self.levels,
                weights_directory= self.weights_directory
            )
        elif self.model_name == 'densenet121':
            self.model = BcnnDensenet121(
                dim_outputs=self.dim_outputs,
                levels = self.levels,
                weights_directory= self.weights_directory
            )    
        else:
            raise ValueError('Unsupported model. Select one of vgg16, resnet50 or densenet121.')  
                           
        self.model.to(self.device)
       
    def train(
        self, hyperparameters: dict, train_loader, val_loader, 
        verbose: bool = True
    ):

        """
        Trains the model instance using the specified data loaders and training parameters.
        Not all training parameters are supported. Sample hyperparameters dictionary:
            HYPERPARAMETERS = {
                'loss_fn': {'type': 'CrossEntropyLoss','alphas':[0.5,0.5], threshold: None}, 
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
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            raise ValueError('Unsupported loss function. Select CrossEntropyLoss.')
        
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
                
                # print gradient norms for coarse head
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
                    
    def gridsearch(self, parameter_grid: dict, train_loader, val_loader, scoring_fn):

        """
        Performs grid search over a set of hyperparameters to identify the best configuration.
        Sampler parameter grid to search:
            HYPERPARAMETER_SEARCH_GRID = {
                'loss_fn': [
                    {'criterion': 'CrossEntropyLoss', 'alpha': [0.5,0.5], 'thresholds': None},
                ],  
                'optimizer': ['Adam'],
                'lr': [1e-3, 5e-4, 1e-4],
                'epochs': [50],
                'scheduler': [
                    {'type': 'StepLR', 'step_size': 10, 'gamma': 0.1},
                ],
                'early_stopping': [
                    {'patience': 10, 'delta': 0.005},
                ],
            }

        Args:
            parameter_grid (dict): Dictionary where keys are hyperparameter names and values are lists of values to try.
            train_loader (DataLoader): PyTorch DataLoader for the training dataset.
            val_loader (DataLoader): PyTorch DataLoader for the validation dataset.
            scoring_fn (callable): A function to evaluate model predictions (e.g., accuracy_score or f1_score).
        """

        parameters = list(parameter_grid.keys())

        best_score = float('-inf')
        best_params = None

        print('\nStarting hyperparameter tuning!')

        start = time.time()
        for iter, param_values in enumerate(product(*parameter_grid.values())):

            current_parameters = dict(zip(parameters, param_values))
            print(f'\nStarting Iteration {iter+1}!')
            print(f'Parameters: {current_parameters}')

            start_iter = time.time()

            # Initiate a new Model object and train based on current parameters
            current_model = BCNN_Model(
                weights_directory = self.weights_directory,
                dim_outputs = self.dim_outputs,
                model_name = self.model_name,
                device = self.device,
                seed = self.seed,
                levels = self.levels
            )

            current_model.train(
                hyperparameters = current_parameters,
                train_loader = train_loader,
                val_loader = val_loader,
                verbose = False
            )

            # Get predictions and score on validation set
            labels, _, preds,_ = current_model.predict(val_loader)

            score = 0
            for l in range(self.levels):
                score += scoring_fn(labels[l].cpu(), preds[l].cpu())
            score = score/self.levels    

            elapsed_iter = time.time() - start_iter
            print(f'Completed Iteration {iter+1}! Time Elapsed {elapsed_iter:.2f} sec. | Score: {score:.4f}')

            # Update parameters if necessary
            if score > best_score:
                best_score = score
                best_params = current_parameters
            
            del current_model
        
        elapsed = time.time() - start
        print(f'\nHyperparameter Tuning Finished! Total Time Elapsed: {elapsed:.2f} sec.')

        return best_params, best_score                
                        
                        
        
                    
                
        
        

        
import os
import random
from torch.utils.data import Dataset, Subset, DataLoader, SequentialSampler, WeightedRandomSampler, random_split
import torch 
import numpy as np 
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from collections import Counter
from extra_functions import set_seed
import copy


class ImageDataset(Dataset):

    """
    A custom PyTorch Dataset for loading and preprocessing image data from a directory
    where each subfolder represents a class.

    This class handles:
    - Class-wise and random sampling with optional size limits
    - Optional image transforms for data augmentation
    - Preprocessing and setup for imbalanced class handling

    Args:
        data_directory (str): Path to the root dataset directory. Each subdirectory should represent a class or another subdirectory to check.
        data_subdirectories (list of str, optional): Subdirectories with additional images, each sub-subdirectory should represent a class.
        class_names (list, optional): List of class names to include. If None, all subdirectories are included.
        class_sizes (list, optional): Number of samples to include per class. If None, uses `max_class_size` for all.
        class_ids (list, optional): Numeric ID for each class (aligned with `class_names`).
        max_class_size (int, optional): Default maximum number of samples to draw per class. Defaults to 10,000.
        image_resolution (int, optional): Final size (height and width) to resize images to. Defaults to 28.
        image_transforms (callable, optional): Image transformations (e.g., data augmentations) to apply. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 666.

    Attributes:
        data_directory (str): Path to the dataset root directory.
        data_subdirectories (list of str): Subdirectories with additional images.
        seed (int): Random seed used for sampling.
        class_names (list): Sorted list of class names included in the dataset.
        class_sizes (torch.Tensor): Tensor of the actual sampled size per class.
        class_ids (list): Numeric ID for each class (aligned with `class_names`).
        image_paths (list): List of file paths to all sampled images.
        labels (list): List of numeric class IDs corresponding to each image.
        image_resolution (int): Size to which each image is resized.
        image_transforms (callable or None): Image transformations applied during training or inference.
        format_file (str): Format of images files. Default: '.tif' 
    """

    def __init__(self, data_directory, data_subdirectories: list = None, class_names: list = None, 
                 class_sizes: list = None, class_ids: list = None, max_class_size: int = 10000, 
                 image_resolution: int = 28, image_transforms = None, seed: int = 666,
                 format_file = '.tif'):
        
        self.data_directory = data_directory
        self.data_subdirectories = ['']
        self.seed = seed
        self.class_names = class_names

        set_seed(seed)

        # Additional subdirectories to check
        if data_subdirectories is not None:
            self.data_subdirectories.extend(data_subdirectories)

        # Specify subset of classes to consider; all classes considered if None
        if class_names is None:
            class_names = sorted(os.listdir(self.data_directory))

        # Specify initial number of samples to consider per class; max if None
        if class_sizes is None:
            class_sizes = [max_class_size] * len(class_names)

        # Specify numeric class ID/index per class; in alphabetical order if None
        if class_ids is None:
            class_ids = list(range(len(self.class_names)))
        
        self.class_names, self.class_sizes, self.class_ids = map(
            list, zip(*sorted(zip(class_names, class_sizes, class_ids)))
        )
               
        # Iterate through each class and sample .tif images only; append paths and labels
        self.image_paths = []
        self.labels = []

        for class_id, class_name in zip(self.class_ids, self.class_names):
            
            # Retrieve all image paths across directories for specified class
            class_paths = []

            for data_subdirectory in self.data_subdirectories:

                class_directory = os.path.join(data_directory, data_subdirectory, class_name)

                if os.path.isdir(class_directory):
                    class_paths.extend(
                        [os.path.join(class_directory, filename) for filename in os.listdir(class_directory)]
                    )

            # Determine new class size and sample images, only include .tif files
            class_idx = class_ids.index(class_id)
            new_class_size = min(self.class_sizes[class_idx], len(class_paths))

            random.seed(self.seed)
            sampled_paths = random.sample(class_paths, new_class_size)
            
            for image_path in sampled_paths:
                if image_path.lower().endswith(format_file):
                    try:
                        with Image.open(image_path) as img:
                            img.verify()
                        self.image_paths.append(image_path)
                    except (UnidentifiedImageError, OSError, ValueError):
                        new_class_size -= 1
                else:
                    new_class_size -= 1

            self.class_sizes[class_idx] = new_class_size
            self.labels.extend([class_id] * new_class_size)
        
        # Other class initializations
        self.image_resolution = image_resolution
        self.image_transforms = image_transforms
        
    
    def __len__(self):

        """
        Returns the number of samples in the Dataset.
        """

        return len(self.image_paths)
    

    def __getitem__(self, idx):

        """
        Returns the image and label of specified sample.

        Args:
            idx (int): Index of specified sample.
        """
        
        image = Image.open(self.image_paths[idx]).convert('L')
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.image_transforms:
            image = self.image_transforms(image)

        image = image.repeat(3, 1, 1)
        
        return image, label
            
    def print_dataset_details(self, indices: list = None, subset_name: str = None):

        """
        Prints the class distribution of the Dataset.

        Args:
            indices (list, optional): Specific indices of subset of Dataset to consider.
            subset_name (str, optional): Specific name of subset of Dataset to print.
        """

        if indices is None:
            filtered_labels = self.labels
        else:
            filtered_labels = [self.labels[i] for i in indices]

        filtered_counts = dict(Counter(filtered_labels))

        if subset_name is None:
            print(f'\nTotal Dataset Size: {len(filtered_labels)}')
        else:
            print(f'\n{subset_name} Dataset Size: {len(filtered_labels)}')

        for class_id, class_name in zip(self.class_ids, self.class_names):
            class_prop = filtered_counts[class_id] / len(filtered_labels)

            print(f'Class Name: {class_name} | Class Label: {class_id} | Count: {filtered_counts[class_id]} ' +
                    f'| Prop: {class_prop:.2f}'
                )
            
    def get_dataset_details(self, indices: list = None):

        """
        Get the class distribution of the Dataset.

        Args:
            indices (list, optional): Specific indices of subset of Dataset to consider.
        Output: 
            Dictionary containing: class name, label, count and proportion relative the whole sample    
        """
        
        all_class_names = []
        all_class_labels = []
        all_counts = []
        all_props = []

        if indices is None:
            filtered_labels = self.labels
        else:
            filtered_labels = [self.labels[i] for i in indices]

        filtered_counts = dict(Counter(filtered_labels))

        for class_id, class_name in zip(self.class_ids, self.class_names):
            class_prop = filtered_counts[class_id] / len(filtered_labels)
            
            all_class_names.append(class_name)
            all_class_labels.append(class_id)
            all_counts.append(filtered_counts[class_id])
            all_props.append(class_prop)
            
        data_details ={
            'Class': all_class_names,
            'Label': all_class_labels,
            'Counts': all_counts,
            'Prop': all_props
            } 
            
        return data_details
    
    def split_train_test_val(self, train_prop: float = 0.7, val_prop: float = 0.1, test_prop: float = 0.2, verbose: bool = True):

        """
        Returns indices corresponding to the train, validation and test subsets of the Dataset.

        Args:
            trian_prop (float): Proportion of samples to allocate to the train subset.
            val_prop (float): Proportion of samples to allocate to the validation subset.
            test_prop (float): Proportion of samples to allocate to the test subset.
            verbose (bool): Specifies whether to print distributions of subsets.
        """

        train_split, val_split, test_split = random_split(
            range(len(self)),
            lengths = [train_prop, val_prop, test_prop],
            generator = torch.Generator().manual_seed(self.seed)
        )

        if verbose:
            self.print_dataset_details(train_split.indices, 'Train')
            self.print_dataset_details(val_split.indices, 'Validation')
            self.print_dataset_details(test_split.indices, 'Test')

        return train_split.indices, val_split.indices, test_split.indices
    
    def append_image_transforms(self, image_transforms: transforms.Compose = None, 
                                replace: bool = False, verbose: bool = False):
        
        """
        Appends image transformations to existing transformation pipeline or replaces.
        If multiple `ToTensor()` transformations are included in the resulting pipeline, only the last instance is kept.
        If there are no `ToTensor()` transformations in the resulting pipeline, it is appended.

        Args:
            image_transforms(transfors.Compose, optional): Iterable of image transformations to append.
            replace (bool): Specifies whether to replace with or append the above image_transforms.
            verbose (bool): Specifies whether to print the resulting image transformation pipeline.
        """
        
        if image_transforms is None:
            if self.image_transforms is None:
                image_transforms_list = []
            else: 
                image_transforms_list = self.image_transforms.transforms
        else:
            if replace:
                image_transforms_list = image_transforms.transforms
            else:
                image_transforms_list = self.image_transforms.transforms + image_transforms.transforms

        image_transforms_cleaned = []
        to_tensor_indices = [i for i, tf in enumerate(image_transforms_list) if isinstance(tf, transforms.ToTensor)]

        if to_tensor_indices:
            last_idx = to_tensor_indices[-1]
            image_transforms_cleaned = [tf for i, tf in enumerate(image_transforms_list) if not isinstance(tf, transforms.ToTensor) or i == last_idx]
        else:
            image_transforms_cleaned = image_transforms_list + [transforms.ToTensor()]

        self.image_transforms = transforms.Compose(image_transforms_cleaned)

        if verbose:
            self.print_image_transforms()
            
    def print_image_transforms(self):

        """
        Prints the ordered image transformations applied to the Dataset.
        """

        print('\nCurrent Image Transform Pipeline:')
        for tf in self.image_transforms.transforms:
            print(' ', tf)    
            
    
    def create_dataloaders(self, batch_size: int, train_indices, val_indices, test_indices,
                           image_transforms: transforms.Compose = None, transform_val: bool = False, 
                           train_sample_weights: torch.tensor = None):
        
        """
        Creates the train, validatinon and test DataLoaders required for training a PyTorch model.
        If `train_sample_weights` is specified, they are supplied to WeightedRandomSampler for the train subset.

        Args:
            batch_size (int): Sizes of batches to process samples in DataLoader.
            train_indices (list): Indices corresponding to the train subset of the Dataset.
            val_indices (list): Indices corresponding to the validation subset of the Dataset.
            test_indices (list): Indices corresponding to the test subset of the Dataset.
            image_transforms (transforms.Compose, optional): Additional image transformations for the train subset.
            transform_val (bool): Specifies whether to apply train image transformations to the validation subset.
            train_sample_weights (torch.tensor, optional): Contains weights for each sample in the train subset.
        """

        if image_transforms is not None:
            dataset_aug = copy.deepcopy(self)
            dataset_aug.append_image_transforms(
                image_transforms = image_transforms, verbose = False
            )
            train_dataset = Subset(dataset_aug, train_indices)
            if transform_val:
                val_dataset = Subset(dataset_aug, val_indices)
            else:
                val_dataset = Subset(self, val_indices)
        else:
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
        
        test_dataset = Subset(self, test_indices)

        if train_sample_weights is None:
            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, 
                generator = torch.Generator().manual_seed(self.seed)
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size = batch_size, 
                sampler = WeightedRandomSampler(train_sample_weights, num_samples = len(train_sample_weights), replacement = True)
            )

        val_loader = DataLoader(
            val_dataset, batch_size = batch_size, sampler = SequentialSampler(val_dataset)
        )
        test_loader = DataLoader(
            test_dataset, batch_size = batch_size, sampler = SequentialSampler(test_dataset)
        )

        return train_loader, val_loader, test_loader            
    
    

def merge_classes(dataset,classes_to_merge_list,new_names_list):
        
    """
    Merge multiple classes within a Dataset.

    Args:
        Dataset (Dataset): An instance of the ImageDataset class.
        classes_to_merge_list (list of str): A list where each sublist contains 
            the class names to merged. To perform multiple merges, provide 
            multiple sublists. 
        new_names_list (list of str): A list of names of newly merged classes. 
    """

    # Create an instance copy
    dataset_copy = copy.deepcopy(dataset)

    for indx, new_name in enumerate(new_names_list):
        dataset_loop = copy.deepcopy(dataset_copy)
        classes_to_merge = classes_to_merge_list[indx]
        
        # If new name already exists
        if (new_name in classes_to_merge): 
            indx_name = dataset_loop.class_names.index(new_name)
            dataset_loop.class_names[indx_name] = new_name + '_old'
            indx_name = classes_to_merge.index(new_name)
            classes_to_merge[indx_name] = new_name + '_old'

        # Get indices involved
        n_categories = len(classes_to_merge)
        indx_order = []
        inds_class = []

        for i in range(n_categories):
            name_idx = dataset_loop.class_names.index(classes_to_merge[i])
            indx_order.append(name_idx)
            inds_class.append(dataset_copy.class_ids[name_idx])
            
        # Select label to keep
        selected_label = min(inds_class)
        selected_order = dataset_loop.class_ids.index(selected_label)
        # class_indices_to_change = [e for e in inds_class if e!=selected_label]
        order_indices_to_change = [e for e in indx_order if e!=selected_order]

        # New names
        dataset_copy.class_names[selected_order] = new_name
        new_class_names = [
            x
            for i, x in enumerate(dataset_copy.class_names) 
            if i not in order_indices_to_change
        ]
        dataset_copy.class_names = new_class_names

        # New indices
        new_ids = list(range(0, len(dataset_copy.class_names)))
        dataset_copy.class_ids = new_ids
                
        # Change labels
        old_np_names = np.array(dataset_loop.class_names)
        new_labels = np.full(len(dataset_copy.labels),None)                
        numpy_old_labels = np.array(dataset_loop.labels)
        for class_id, name in zip(dataset_copy.class_ids, dataset_copy.class_names):
            if name in dataset_loop.class_names:
                idx = np.where(old_np_names == name)[0][0]
                old_label = dataset_loop.class_ids[idx]
                new_label = class_id
                idx_label = np.where(numpy_old_labels == old_label)
                new_labels[idx_label] = new_label
            else:
                idx = np.where(np.isin(old_np_names ,classes_to_merge))[0].tolist()
                old_labels = [dataset_loop.class_ids[i] for i in idx]
                idx_label = np.where(np.isin(numpy_old_labels,old_labels))
                new_labels[idx_label] = class_id

        dataset_copy.labels = new_labels.tolist()

        # Change Sizes
        sizes = dict(Counter(dataset_copy.labels))
        new_sizes = []
        for class_id in dataset_copy.class_ids:
            new_sizes.append(sizes[class_id])
            
        dataset_copy.class_sizes = new_sizes

        dataset_copy.class_names, dataset_copy.class_sizes, dataset_copy.class_ids = map(
                    list, zip(*sorted(zip(dataset_copy.class_names, 
                                        dataset_copy.class_sizes, 
                                        dataset_copy.class_ids)))
                )

    return dataset_copy




class HierImageDataset(Dataset):
    
    """
    A custom PyTorch Dataset for creating a dataset with hierarchical labels

    This class handles:
    - Optional image transforms for data augmentation

    Args:
        base_dataset (ImageDataset): The original dataset from which the hierarchical dataset will be built. 
        groups (list of str): The names of the classes in `base_dataset` that will be combined into groups
        coarse_names (list of str): The names assigned to the newly formed coarse groups. 
        levels (int): The number of levels to include in the hierarchy. (IMPORTANT: For now only working with 2)
        image_transforms (callable, optional): Image transformations (e.g., data augmentations) to apply. Defaults to None.

        
    Attributes:
        data_directory (str): Path to the dataset root directory (from base_dataset).
        data_subdirectories (list of str): Subdirectories with additional images (from base_dataset).
        seed (int): Random seed used for sampling (from base_dataset).
        class_names (list): List by level with sorted class names included in the dataset.
        class_sizes (torch.Tensor): List by level with the actual sampled size per class.
        class_ids (list): List by level with numeric ID for each class (aligned with `class_names`).
        image_paths (list): List of file paths to all sampled images (from base_dataset).
        labels (list): List by level with numeric class IDs corresponding to each image.
        image_resolution (int): Size to which each image is resized (from base_dataset).
        image_transforms (callable or None): Image transformations applied during training or inference.
        format_file (str): Format of images files. Default: '.tif (from base_dataset)' 
        
    """
       
    def __init__(self, base_dataset,groups,coarse_names,levels=2, 
                 image_transforms = None
        ):
        
        self.data_directory = base_dataset.data_directory
        self.data_subdirectories = base_dataset.data_subdirectories
        self.seed = base_dataset.seed
        self.image_resolution =  base_dataset.image_resolution
        self.image_paths = base_dataset.image_paths
        self.levels = levels # for now two 

        
        # Create hierarchies
        dataset_copy = copy.deepcopy(base_dataset)
        numpy_old_labels = np.array(dataset_copy.labels)
        coarse_labels = np.full(len(dataset_copy.labels),None)   
        coarse_sizes = []
                        
        for g in range(len(groups)):
            curr_coarse_label = g
            names = groups[g]
            for name in names: 
                curr_label = base_dataset.class_names.index(name)
                idx_label = np.where(numpy_old_labels == curr_label)
                coarse_labels[idx_label] = curr_coarse_label
            coarse_sizes.append(sum(coarse_labels==curr_coarse_label))

        self.labels = [coarse_labels.tolist(),dataset_copy.labels]
        self.class_names = [coarse_names,dataset_copy.class_names]
        self.class_sizes = [coarse_sizes, dataset_copy.class_sizes]
        self.class_ids = [list(range(len(groups))),dataset_copy.class_ids]
        
        
        # Other class initializations
        self.image_resolution = base_dataset.image_resolution
        self.image_transforms = image_transforms
        
        
    def __len__(self):

        """
        Returns the number of samples in the Dataset.
        """

        return len(self.image_paths)
    
    def __getitem__(self, idx):

        """
        Returns the image and labels of specified sample.

        Args:
            idx (int): Index of specified sample.
        """
        
        image = Image.open(self.image_paths[idx]).convert('L')
        labels = tuple(
            torch.tensor(self.labels[i][idx], dtype=torch.long)
            for i in range(self.levels)
        )
        if self.image_transforms:
            image = self.image_transforms(image)

        image = image.repeat(3, 1, 1)
        
        return image, labels
    
    
    def print_dataset_details(self):

        """
        Prints the class distribution of the Dataset.

        """

        print(f'\nTotal Dataset: Size = {len(self)}| Levels = {self.levels}')
        
        for l in range(self.levels):
            level_counts = dict(Counter(self.labels[l]))
            for class_id, class_name in zip(self.class_ids[l], self.class_names[l]):
                class_prop = level_counts[class_id] / len(self)

                print(
                    f'Level: {l} | Class Name: {class_name} | Class Label: {class_id}' +
                    f'| Count: {level_counts[class_id]} | Prop: {class_prop:.2f}'
                )
                
    
    def split_train_test_val(
        self, train_prop: float = 0.7, val_prop: float = 0.1, test_prop: float = 0.2
    ):

        """
        Returns indices corresponding to the train, validation and test subsets of the Dataset.

        Args:
            trian_prop (float): Proportion of samples to allocate to the train subset.
            val_prop (float): Proportion of samples to allocate to the validation subset.
            test_prop (float): Proportion of samples to allocate to the test subset.
        """

        train_split, val_split, test_split = random_split(
            range(len(self)),
            lengths = [train_prop, val_prop, test_prop],
            generator = torch.Generator().manual_seed(self.seed)
        )

        return train_split.indices, val_split.indices, test_split.indices
    
    def append_image_transforms(
        self, image_transforms: transforms.Compose = None, replace: bool = False
        ):        
        """
        Appends image transformations to existing transformation pipeline or replaces.
        If multiple `ToTensor()` transformations are included in the resulting pipeline, only the last instance is kept.
        If there are no `ToTensor()` transformations in the resulting pipeline, it is appended.

        Args:
            image_transforms(transfors.Compose, optional): Iterable of image transformations to append.
            replace (bool): Specifies whether to replace with or append the above image_transforms.
        """
        
        if image_transforms is None:
            if self.image_transforms is None:
                image_transforms_list = []
            else: 
                image_transforms_list = self.image_transforms.transforms
        else:
            if replace:
                image_transforms_list = image_transforms.transforms
            else:
                image_transforms_list = self.image_transforms.transforms + image_transforms.transforms

        image_transforms_cleaned = []
        to_tensor_indices = [i for i, tf in enumerate(image_transforms_list) if isinstance(tf, transforms.ToTensor)]

        if to_tensor_indices:
            last_idx = to_tensor_indices[-1]
            image_transforms_cleaned = [tf for i, tf in enumerate(image_transforms_list) if not isinstance(tf, transforms.ToTensor) or i == last_idx]
        else:
            image_transforms_cleaned = image_transforms_list + [transforms.ToTensor()]

        self.image_transforms = transforms.Compose(image_transforms_cleaned)
                    
    def create_dataloaders(
        self, batch_size: int, train_indices, val_indices, test_indices,
        image_transforms: transforms.Compose = None, transform_val: bool = False, 
        train_sample_weights: torch.tensor = None
    ):
        
        """
        Creates the train, validatinon and test DataLoaders required for training a PyTorch model.
        If `train_sample_weights` is specified, they are supplied to WeightedRandomSampler for the train subset.

        Args:
            batch_size (int): Sizes of batches to process samples in DataLoader.
            train_indices (list): Indices corresponding to the train subset of the Dataset.
            val_indices (list): Indices corresponding to the validation subset of the Dataset.
            test_indices (list): Indices corresponding to the test subset of the Dataset.
            image_transforms (transforms.Compose, optional): Additional image transformations for the train subset.
            transform_val (bool): Specifies whether to apply train image transformations to the validation subset.
            train_sample_weights (torch.tensor, optional): Contains weights for each sample in the train subset.
        """

        if image_transforms is not None:
            dataset_aug = copy.deepcopy(self)
            dataset_aug.append_image_transforms(
                image_transforms = image_transforms, verbose = False
            )
            train_dataset = Subset(dataset_aug, train_indices)
            if transform_val:
                val_dataset = Subset(dataset_aug, val_indices)
            else:
                val_dataset = Subset(self, val_indices)
        else:
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
        
        test_dataset = Subset(self, test_indices)

        if train_sample_weights is None:
            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, 
                generator = torch.Generator().manual_seed(self.seed)
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size = batch_size, 
                sampler = WeightedRandomSampler(train_sample_weights, num_samples = len(train_sample_weights), replacement = True)
            )

        val_loader = DataLoader(
            val_dataset, batch_size = batch_size, sampler = SequentialSampler(val_dataset)
        )
        test_loader = DataLoader(
            test_dataset, batch_size = batch_size, sampler = SequentialSampler(test_dataset)
        )

        return train_loader, val_loader, test_loader   
        
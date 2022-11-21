import torch
import h5py
import pandas as pd
import numpy as np
import torchvision

class H5Dataset(torch.utils.data.Dataset):
    # Given a list of h5_path files, extract the voltage, distribution, and loss values
    """
    h5_path: directory containing all hdf5 files
    transform: If you want to transform the data into a different size or not. Defult is original image size
    size: how many pixel wide do you want your image to be?
    """
    def __init__(self, h5_paths, transform=False, size = 94):
        self.h5_paths = h5_paths
        self.transform=transform
        self.size = size
        # Code to get data from hdf5 files and put into memory
        i=0
        self.voltages, distro, output= self.extract_data(h5_paths[i])
        self.distro_0 = distro[0]
        self.distro_1 = distro[1]
        self.distro_2 = distro[2]
        self.distro_3 = distro[3]
        self.distro_4 = distro[4]
        self.output_0 = output[0]
        self.output_1 = output[1]
        self.output_2 = output[2]
        self.output_3 = output[3]
        print(f"Added first file, current memory of dataset: {self.get_memory()}")
        i+=1
        while(i<len(h5_paths)):
            if h5_paths[1].endswith('.hdf5'):
                a, b, c = self.extract_data(h5_paths[i])
                self.voltages = np.concatenate((self.voltages,a))
                self.distro_0 = np.concatenate((self.distro_0,b[0]))
                self.distro_1 = np.concatenate((self.distro_1,b[1]))
                self.distro_2 = np.concatenate((self.distro_2,b[2]))
                self.distro_3 = np.concatenate((self.distro_3,b[3]))
                self.distro_4 = np.concatenate((self.distro_4,b[4]))
                self.output_0 = np.concatenate((self.output_0,c[0]))
                self.output_1 = np.concatenate((self.output_1,c[1]))
                self.output_2 = np.concatenate((self.output_2,c[2]))
                self.output_3 = np.concatenate((self.output_3,c[3]))
            i+=1
            print(f"Added another file, current memory of dataset: {self.get_memory()}")
        # Make sure all particles that starts are inside the aperature
        distro_sum = self.distro_0.sum(axis=(1,2,3))  # TODO: Change to 0!!
        mask = distro_sum>60000  # TODO Also change back to 60000!!
        self.voltages = self.voltages[mask]
        self.distro_0 = self.distro_0[mask]
        self.distro_1 = self.distro_1[mask]
        self.distro_2 = self.distro_2[mask]
        self.distro_3 = self.distro_3[mask]
        self.distro_4 = self.distro_4[mask]
        self.output_0 = self.output_0[mask]
        self.output_1 = self.output_1[mask]
        self.output_2 = self.output_2[mask]
        self.output_3 = self.output_3[mask]
        # normalizing the data from [-1,1]
        self.voltages = self.voltages/8
        self.distro_0 = self.distro_0/1  # Small normalizing distro for now
        self.distro_1 = self.distro_1/1
        self.distro_2 = self.distro_2/1
        self.distro_3 = self.distro_3/1
        self.distro_4 = self.distro_4/1
        self.output_0 = self.output_0/1
        self.output_1 = self.output_1/1
        self.output_2 = self.output_2/1
        self.output_3 = self.output_3/1
        self.len = len(self.output_0)
        
        self.resize = torchvision.transforms.Resize((self.size,self.size))

    def __getitem__(self, index):
        if (self.transform):
            return ((self.voltages[index],  # voltages on quad
                     self.Resize(self.distro_0[index]),  # Distribution images
                     self.Resize(self.distro_1[index]),
                     self.Resize(self.distro_2[index]),
                     self.Resize(self.distro_3[index]),
                     self.Resize(self.distro_4[index])),
                    (self.output_0[index],  # outputs
                     self.output_1[index],
                     self.output_2[index],
                     self.output_3[index]))
        else:
            return ((self.voltages[index],  # voltages on quad
                     self.distro_0[index],  # Distribution images
                     self.distro_1[index],
                     self.distro_2[index],
                     self.distro_3[index],
                     self.distro_4[index]),
                    (self.output_0[index],  # outputs
                     self.output_1[index],
                     self.output_2[index],
                     self.output_3[index]))

    def __len__(self):
        return self.len
    
    def Resize(self, data):
        s = np.array(data).shape
        tempdata = torch.zeros((self.size, self.size, s[-1]))
        for i in range(s[-1]):
            tempdata[:,:,i] = self.resize(torch.from_numpy(data[:,:,i]).squeeze().unsqueeze(dim=0))
        return np.array(tempdata)
    
    def get_memory(self):
        """
        Return memory in GBs
        """
        memory = self.voltages.nbytes
        memory += self.distro_0.nbytes
        memory += self.distro_1.nbytes
        memory += self.distro_2.nbytes
        memory += self.distro_3.nbytes
        memory += self.distro_4.nbytes
        memory += self.output_0.nbytes
        memory += self.output_1.nbytes
        memory += self.output_2.nbytes
        memory += self.output_3.nbytes # float32 has 4 byte
        memory = memory/1028**3
        return memory
    
    def extract_data(self, file):
        with h5py.File(file, 'r') as f:
            v = pd.DataFrame(f['Input/v1'])
            nonzero = (v.shape[0]-(v==0).sum())[0]  # Get number of nonzero values from file
            print(nonzero)
            v1 = f['Input/v1'][:nonzero]
            v2 = f['Input/v2'][:nonzero]
            v3 = f['Input/v3'][:nonzero]
            v4 = f['Input/v4'][:nonzero]
            v5 = f['Input/v5'][:nonzero]
            v6 = f['Input/v6'][:nonzero]
            v = pd.DataFrame([v1,v2,v3,v4,v5,v6]).T  # This creates a vector of voltage data

            exit_left = [None]*4
            exit_left[0] = f['07_drift/#of_part_left'][:nonzero]  #Get number of particles after first quad pair
            exit_left[1] = f['10_drift/#of_part_left'][:nonzero]  #Get number of particles after second quad pair
            exit_left[2] = f['13_drift/#of_part_left'][:nonzero]  #Get number of particles after third quad pair
            exit_left[3] = f['17_drift/#of_part_left'][:nonzero]  #Get number of particles left at the last element
            ploss_0 = pd.DataFrame(exit_left[0]) 
            ploss_1 = pd.DataFrame(exit_left[1]) 
            ploss_2 = pd.DataFrame(exit_left[2]) 
            ploss_3 = pd.DataFrame(exit_left[3]) 

            distro = [None]*5
            distro[0] = f['Input/33x33_0'][:nonzero,:,:,:]
            distro[1] = f['Input/33x33_1'][:nonzero,:,:,:]
            distro[2] = f['Input/33x33_2'][:nonzero,:,:,:]
            distro[3] = f['Input/33x33_3'][:nonzero,:,:,:]
            distro[4] = f['Input/33x33_4'][:nonzero,:,:,:]
            print(distro[0].shape)

        voltage = v.values
        distro = np.array(distro)
        loss = np.array(exit_left)
        return (voltage, distro, loss)

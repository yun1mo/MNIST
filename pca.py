import numpy as np
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt

def comb_img(data,col,row):
    new=[]
    for i in range(row):
        for j in range(45):
            a=[]
            for k in range(col):
                a.extend(data[i*col+k][j])
            new.append(a)
    return np.array(new)

def pca(data,n):
    num,dim = data.shape
    mean_val = data.mean(axis=0)
    mean_removed =  data-mean_val
    cov = np.cov(mean_removed,rowvar=0)
    eig_vals,eig_vects = np.linalg.eig(np.mat(cov))
    eig_vals_index = np.argsort(eig_vals)
    eig_vals_index = eig_vals_index[:-(n + 1) : -1]
    reg_eig_vects = eig_vects[:, eig_vals_index]
    low_d_data_mat = mean_removed * reg_eig_vects
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_val
    return np.array(low_d_data_mat),np.array(recon_mat)
    
if __name__ == "__main__":
    
    data_num = 60000 #The number of figures
    fig_w = 45       #width of each figure

    data = np.fromfile("train/mnist_train_data",dtype=np.uint8)
    label = np.fromfile("train/mnist_train_label",dtype=np.uint8)
    
    test = np.fromfile("test/mnist_test_data",dtype=np.uint8)
    
    print(data.shape)
    print(label.shape)
    print test.shape
    #reshape the matrix
    data=data.reshape(60000,45,45)
    com = comb_img(data,5,2)
    plt.axis('off')
    plt.imshow(com,cmap='jet')
    plt.savefig("pic/example.png")
    data = data.reshape(60000,2025)
    test = test.reshape(10000,2025)
    
    final_data = np.concatenate((data,test),axis=0)
    print final_data.shape
    print("After reshape:",data.shape)
    
    dim_list=[50,100,150,200,250]
    for i in dim_list:     
        low_d_data,recon_data = pca(final_data,i)
        print low_d_data.shape
        np.save("train_pca/"+str(i)+"_train.npy",low_d_data[:60000])
        np.save("test_pca/"+str(i)+"_test.npy",low_d_data[60000:])
        
        recon_data = recon_data[:60000].reshape(60000,45,45)
        re_com = comb_img(recon_data,5,2)
        plt.figure()
        plt.axis('off')
        plt.imshow(re_com,cmap='jet')
        plt.savefig("pic/recon_"+str(i)+".png")
    

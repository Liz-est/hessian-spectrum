import re
import numpy as np
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional
import matplotlib.pyplot as plt
from contextlib import nullcontext
#import ipdb
import json
from tqdm import tqdm
import seaborn as sns
from scipy.spatial.distance import jensenshannon


class Hessian(object):
    def __init__(self, model = None,  m = 100, sigma = 1e-5**0.5, ckpt_iteration= 0, train_data = [], block_size = None, batch_size = None, num_v = 10, ctx =nullcontext(), use_minibatch = True, gradient_accumulation_steps = 1, device = 'cuda',sample_layer = None,  ddp = False, comment = None):
        self.model = model
        self.m = m # number of lanzcos basis
        self.sigma = sigma # the standard deviation of gaussian r.v.
        self.ckpt_iteration = ckpt_iteration
        self.train_data = train_data
        self.block_size = block_size
        self.batch_size = batch_size
        self.ctx = ctx
        self.use_minibatch = use_minibatch
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.sample_layer = sample_layer
        self.ddp = ddp
        self.num_v = num_v
        self.num_bins = 1000


        self.num_batches = len(self.train_data)
        self.criterion = nn.CrossEntropyLoss().cuda()
      
        
        print('total batch', self.num_batches)


        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f'name = {name}, #params = {param.numel()}')
        
        
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #n_params = sum(p.numel() for p in self.parameters())
        print('total params', self.total_params)

        

        self.comment = comment + '_minibatch_'+str(self.use_minibatch) +'_bs_'+str(self.batch_size*(self.gradient_accumulation_steps))+ '_m_'+str(self.m)  + '_v_' +str(self.num_v) + '_ckpt_'+str(self.ckpt_iteration)

        self.file_dir = 'files/' + 'normal_init/' + str(self.comment) + '/'
        
        os.makedirs(self.file_dir, exist_ok= True)


    def get_spectrum(self, layer_by_layer = False):
        if layer_by_layer: 
            self.get_spectrum_layer_by_layer()
        else: 
            self.get_spectrum_full()


    def get_spectrum_layer_by_layer(self):
      
        weights_dic, values_dic = {}, {}

        for name, param in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if param.requires_grad:
                print("name computing", name)
                zeros = np.zeros((self.num_v, self.m))
                weights_dic[name] = [row.tolist() for row in zeros]
                values_dic[name] =  [row.tolist() for row in zeros]

   
        t_s = time.time()
        for k in range(self.num_v): 
            print('current k' , k)

            'wiki version'
            T_dic = self.tridiagonalize_by_lanzcos_layer_by_layer(k) #returns a dic: {'name': T}
            
            for name, T in T_dic.items():
                eigenvalues, U  = np.linalg.eigh(T)
                values_dic[name][k] = eigenvalues.tolist()
                weights_dic[name][k] = (U[0]**2).tolist()

            'we also save the inter-medium results'
            self.save_curve(total_time= time.time() - t_s, weights_layer = weights_dic, values_layer = values_dic)

        total_time = time.time() - t_s

        self.save_curve(total_time= total_time, weights_layer = weights_dic, values_layer = values_dic)



    def get_spectrum_full(self):
        weights = np.zeros((self.num_v, self.m))
        values = np.zeros((self.num_v, self.m))
        time_initial = time.time()
      

        for k in range(self.num_v): 
            print('current k' , k)
     

            'wiki version'
            T = self.tridiagonalize_by_lanzcos(k)
            eigenvalues, U  = np.linalg.eigh(T)
            values[k,:] = eigenvalues
            weights[k,:] = U[0]**2
   

            self.save_curve(total_time = time.time() -time_initial, weights_full =  {'weights': weights}, values_full = {'values': values}, grid = [], curve = [])
            
        total_time = time.time() -time_initial
        grid, curve = self.interpolate(weights, values)

        self.save_curve(total_time = total_time, weights_full =  {'weights': weights}, values_full = {'values': values}, grid = grid, curve = curve)

    def save_curve(self,total_time = None, weights_layer = None, values_layer = None, weights_full = None, values_full = None, grid = [], curve = []):

        if total_time != None:         
            file_name = self.file_dir + 'time.txt'
            with open(file_name, "w") as file:
                file.write(str(total_time) + "\n")

        if weights_layer != None:
            weights_layer = {key: weights_layer[key] for key in weights_layer} # convert the values to list
            file_name = self.file_dir + 'weights_layer.json'
            with open(file_name, 'w') as json_file:
                json.dump(weights_layer, json_file)
        

        if values_layer != None:
            values_layer = {key: values_layer[key] for key in values_layer} # convert the values to list
            file_name = self.file_dir + 'values_layer.json'
            with open(file_name, 'w') as json_file:
                json.dump(values_layer, json_file)

        if weights_full != None:
            weights_full = {key: weights_full[key].tolist() for key in weights_full} # convert the values to list
            file_name = self.file_dir + 'weights_full.json'
            with open(file_name, 'w') as json_file:
                json.dump(weights_full, json_file)
        

        if values_full != None:
            values_full = {key: values_full[key].tolist() for key in values_full} # convert the values to list
            file_name = self.file_dir + 'values_full.json'
            with open(file_name, 'w') as json_file:
                json.dump(values_full, json_file)


        if len(grid) != 0: 
            file_name = self.file_dir+ 'grid.txt'
            with open(file_name, "w") as file:
                for item in grid:
                    file.write(str(item) + "\n")

        if len(curve) != 0:
            file_name =  self.file_dir + 'curve.txt'
            with open(file_name, "w") as file:
                for item in curve:
                    file.write(str(item) + "\n")
        

    def load_curve(self, layer_by_layer = False):
        if layer_by_layer: 
            self.load_curve_layer_by_layer()
        else: 
            self.load_curve_full()

      
    def load_curve_layer_by_layer(self):

        'load weights and values:'
        file_name = self.file_dir + 'weights_layer.json'
        with open(file_name, 'r') as json_file:
            weights_dic = json.load(json_file)
        weights_dic = {key: np.array(value) for key, value in weights_dic.items()}

        file_name = self.file_dir + 'values_layer.json'
        with open(file_name, 'r') as json_file:
            values_dic = json.load(json_file)
        values_dic = {key: np.array(value) for key, value in values_dic.items()}

        for name in weights_dic.keys():
            weights = weights_dic[name]
            values = values_dic[name]
            grid, curve = self.interpolate(weights, values)

            print('curve',curve)

            'plot'
            plt.figure()

            plt.plot(grid, curve, label = 'approximated curve', alpha = 0.5)
            plt.xlabel('Eigenvalues')
            plt.ylabel('Frequency')
            #plt.ylim([0,1])
            #plt.xlim([-1, 1])
            plt.legend()
            plt.title(f'model at interation {self.ckpt_iteration}')
            plt.savefig(self.file_dir+'spectrum_'+name+'.png')
            plt.close()

            'log plot'
            plt.figure()
            plt.semilogy(grid, curve, label = 'approximated curve', alpha = 0.5)
            plt.xlabel('Eigenvalues')
            plt.ylabel('Frequency (log)')
            plt.ylim([1e-10,1e2])
            #plt.xlim([3, 5])
            plt.legend()
            plt.title(f'model at interation {self.ckpt_iteration}')
            plt.savefig(self.file_dir+'/spectrum_log_'+name+'.png')
            plt.close()


    def load_curve_full(self):
        'load curve'
        grid = []
        file_name = self.file_dir + 'grid.txt'
        with open(file_name, "r") as file:
            for line in file:
                grid.append(float(line.strip()))  # Use strip() to remove 

        file_name =  self.file_dir + 'curve.txt'
        curve = []
        with open(file_name, "r") as file:
            for line in file:
                curve.append(float(line.strip()))  # Use strip() to remove 

        'plot'
        plt.figure()
        plt.plot(grid, curve, label = 'approximated curve', alpha = 0.5)
        plt.xlabel('Eigenvalues')
        plt.ylabel('Frequency')
        plt.ylim([1e-10,1e2])
        plt.legend()
        plt.title(f'model at interation {self.ckpt_iteration}')
        plt.savefig(self.file_dir+'/spectrum_full_hessian.png')
        plt.close()

        'log plot'
        plt.figure()
        plt.semilogy(grid, curve, label = 'approximated curve', alpha = 0.5)
        plt.xlabel('Eigenvalues')
        plt.ylabel('Frequency (log)')
        plt.ylim([1e-10,1e2])
        plt.legend()
        plt.title(f'model at interation {self.ckpt_iteration}')
        plt.savefig(self.file_dir+'/spectrum_log_full_hessian.png')
        plt.close()


    def tridiagonalize_by_lanzcos_layer_by_layer(self, k):
        v_dic = {} # value: list
        alpha_dic = {} # value: scaler
        w_dic = {} # value: #parameters*1 tensor
        beta_dic = {} # value: scaler
        T_dic = {} # value: m*m tensor 
        'initialize'
        for name, params in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if params.requires_grad:
                v = torch.randn_like(params, dtype = torch.float64) 
                v /= torch.norm(v)
                v_dic[name] = [v.cpu()]
                T_dic[name] = np.zeros((self.m, self.m), dtype= np.float64)


     
        w_prime_dic = self.hessian_vector_product_with_dic_input_image(v_dic, k,0)  
       
        'orthogonalize wprime'
        for name in T_dic.keys():
            alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
            w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1]
            T_dic[name][0, 0] = alpha_dic[name] 

        'iteration'
        print('runing lanczos')
        for j in range(1, self.m):

            for name in T_dic.keys(): 
                beta = torch.norm(w_dic[name])
                beta_dic[name] = beta
                if beta >1e-8:
                    v_dic[name].append( w_dic[name] / beta )
                else:
                    #print('The value of beta is 0')
                    v_dic[name].append( w_dic[name] / 1e-8 )
                    #raise ZeroDivisionError('The value of beta is 0')
                if len(v_dic[name]) > 2:
                    del v_dic[name][0]  # keep this list short to save memory

            t_hessian = time.time()
          
            w_prime_dic = self.hessian_vector_product_with_dic_input_image(v_dic, k,j)  
           
            print('t for hessian', time.time() - t_hessian)
            
            #print("w_prime_dic", w_prime_dic)

            'orthogonalize wprime'
            for name in T_dic.keys():
                alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
                w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1] - beta_dic[name] * v_dic[name][-2]
                T_dic[name][j, j] = alpha_dic[name] 
                T_dic[name][j-1, j ] = beta_dic[name] 
                T_dic[name][j , j-1] = beta_dic[name]

        #print("T_dic", T_dic)
        return T_dic


    def tridiagonalize_by_lanzcos(self, k):
        'set up'
        v_list = []
        T = np.zeros((self.m, self.m), dtype= np.float64)

        'initialization'
        v = torch.randn(self.total_params, dtype = torch.float64) 
        v /= torch.norm(v)
        v_list.append(v.cpu())


        w_prime = self.hessian_vector_product_with_tensor_input_image(v_list[-1], k,0)
        
        'orthogonalize wprime'
        alpha = torch.sum(w_prime * v_list[-1])
        w = w_prime - alpha * v_list[-1]
        T[0, 0] = alpha

        'iteration'
        #t_s = time.time()
        print('runing lanczos')
        for j in range(1, self.m):
            beta = torch.norm(w)
            if beta >1e-8:
                v_list.append(w / beta)

            else:
                v_list.append(w / 1e-8)

                # print(f' since beta = {beta}, generate v that orthogonal to all previous v')
                # # Generate a random vector orthogonal to previous ones
                # v = torch.randn(self.total_params) *(1/self.total_params)**0.5
                # for i in range(j):
                #     vi = v_list[i]
                #     v -= torch.sum(vi * v) * vi
                # v /= torch.norm(v)
                if len(v_list) > 2:
                    del v_list[0]  # keep this list short to save memory

           
            w_prime = self.hessian_vector_product_with_tensor_input_image(v_list[-1], k,j)
            
            alpha = torch.sum(w_prime* v_list[-1])
            w = w_prime - alpha * v_list[-1] - beta * v_list[-2]
            T[j, j] = alpha
            T[j-1, j ] = beta
            T[j , j-1] = beta

        return  T


    def interpolate(self,weights, values):
        left_boundary = np.mean(np.min(values, axis = 1))-1
        right_boundary= np.mean(np.max(values, axis = 1)) +1
        n_grid = 50000
        grid = np.linspace(left_boundary, right_boundary, n_grid).tolist()
        density_all = np.zeros((self.num_v, n_grid))

        for k  in range(self.num_v):
            for idx, t  in enumerate(grid):
                values_each_v_t = self.gaussian_density(t, values[k,:])
                density_each_v_t = np.sum(values_each_v_t * weights[k,:])
                density_all[k,idx] = density_each_v_t

        density_avg = np.nanmean(density_all, axis = 0)
        norm_fact = np.sum(density_avg)*(grid[1]- grid[0])
        density_avg /= norm_fact

        return grid, density_avg
    
    
    def hessian_vector_product_with_dic_input_image(self, d_dic, v_step, l_step):

        'comput hessian_vector product, takes a dictionary as input, the values of dic is a list of historical lanscoz directions: d_dic = {name, [history v..]}'
        self.model.eval()
        self.model.zero_grad(set_to_none = True)

        time_initialize = time.time()
        'initialize'
        hd_dic = {}
        for name, param in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if param.requires_grad:
                hd_dic[name]  = torch.zeros_like(param.data).cpu()

        #print('initial ', time.time()  - time_initialize)
        time_getdata = time.time()
        t_hd = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.train_data):
            #print('b indx', batch_idx, 'get data', time.time() - time_getdata)

            X = inputs.cuda()
            Y = targets.cuda()

            #print('X norm', torch.norm(X))
            
            outputs = self.model(X)
            loss = self.criterion(outputs, Y)
            loss.backward(create_graph= True)
            g_dic = {}
            for name, param in self.model.named_parameters():
                if name not in self.sample_layer:
                    continue
                if param.requires_grad:
                    g = param.grad.double()
                    #print(f"name: {name}; grad: {g}")
                    g_dic[name] = g
  
            self.model.zero_grad(set_to_none = True)
            for name, param in self.model.named_parameters():
                if name not in self.sample_layer:
                    continue
                if param.requires_grad:
                    l = torch.sum(g_dic[name].cuda() * d_dic[name][-1].cuda())
                    l.backward(retain_graph = True)
                    hd = param.grad.double().data.clone()

                    hd_dic[name]  += hd.cpu()
                    self.model.zero_grad(set_to_none = True)

            if batch_idx % 10 == 1 or batch_idx == self.gradient_accumulation_steps-1:
                print(f'layer hessian: load_iter ={self.ckpt_iteration}, current random direction = {v_step} / {self.num_v}, lanczos step = {l_step} / {self.m}, Hd current batch = {batch_idx} / {self.num_batches}, time = {time.time() -t_hd}')
                t_hd = time.time()
                #print('hd_dic', hd_dic)
            if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
                break
        return hd_dic


    def hessian_vector_product_with_tensor_input_image(self, d_tensor, v_step, l_step):
        'comput hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) )'

        d_tensor = d_tensor.cuda()
        self.model.eval()
        self.model.zero_grad(set_to_none = True)
        total_hd_tensor = 0

        t_hd = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.train_data):
            #print('b indx', batch_idx, 'get data', time.time() - time_getdata)

            X = inputs.cuda()
            Y = targets.cuda()
            outputs = self.model(X)
            loss = self.criterion(outputs, Y)

            loss.backward(create_graph= True)
            g_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_list.append(torch.flatten(param.grad.double()))

            g_tensor = torch.cat(g_list, dim = 0)
            
            self.model.zero_grad(set_to_none = True)
            g_tensor = g_tensor.cuda()
            l = torch.sum(g_tensor*d_tensor)
            l.backward(retain_graph = True)

            hd_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    hd_list.append(torch.flatten(param.grad.double().data.clone()))

            hd_tensor = torch.cat(hd_list, dim = 0)
            self.model.zero_grad(set_to_none = True)
            hd_tensor = hd_tensor.cpu()
            total_hd_tensor += hd_tensor

            if batch_idx % 10 == 1 or batch_idx == self.gradient_accumulation_steps-1:
                print(f'full hessian: load_iter ={self.ckpt_iteration} current random direction = {v_step} / {self.num_v}, lanczos step = {l_step} / {self.m}, Hd current batch = {batch_idx} / {self.num_batches}, time = {time.time() -t_hd}')
                t_hd = time.time()

            if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
                break
        return total_hd_tensor
    
    
    def get_full_hessian(self):
        self.model.eval()
        self.model.zero_grad(set_to_none = True)

        def hessian_calculation(g_tensor, batch_idx):
            g_tensor = g_tensor.cuda()
            total_params = g_tensor.size(0)
            hessian_list = []
            t_d = time.time()
            for d in range(total_params):
                unit_vector = torch.zeros(total_params)
                unit_vector[d] = 1
                unit_vector = unit_vector.cuda()
                l = torch.sum(g_tensor * unit_vector)
                l.backward(retain_graph= True)

                hessian_row = []
                for name, param in self.model.named_parameters():
                    if 'ln' in name or 'bias' in name or 'wte' in name or 'wpe' in name:
                        continue
                    if param.requires_grad:
                        #print('name',name, param.grad)
                        hessian_row.append(param.grad.double().data.clone())
                
                self.model.zero_grad(set_to_none = True)
                hessian_row = [g.flatten() for g in hessian_row] 
                hessian_row = [g.cpu() for g in hessian_row]
                hessian_row = torch.cat(hessian_row)
                #print('hessian_row', hessian_row)   
                hessian_list.append(hessian_row)
                # if d % 1000 == 0:
                #     print(f'Computing hessian: current batch = {batch_idx}/{self.num_batches}, current row of a hessian: {d}/{total_params}, total time = {time.time()- t_d} ')

            hessian = torch.stack(hessian_list, dim = 1)

            #print('hessian', hessian)   
            return hessian

        full_hessian = 0

        for batch_idx, data in enumerate(self.train_data):

            X_train = data['X_train']
            Y_train = data['Y_train']

            output = self.model(X_train)

            #loss = F.cross_entropy(output, Y_train) 
            loss =self.loss(output, Y_train)

            loss.backward(create_graph= True)

            g_list = []
            count = 0
            for name, param in self.model.named_parameters():
                #if 'ln' in name or 'bias' in name:
                if 'ln' in name or 'bias' in name or 'wte' in name or 'wpe' in name:
                    continue
                if param.requires_grad:
                    count += param.numel()
                    #print('g shape', param.grad , param.grad.shape)
                    g_list.append(torch.flatten(param.grad.double()))
                    #print('name',name, g_list[-1].size())

            g_tensor = torch.cat(g_list, dim = 0)
            #print('g_tensor',g_tensor)
            self.model.zero_grad(set_to_none = True)
            H = hessian_calculation(g_tensor, batch_idx)
            full_hessian += H


        full_hessian = torch.nan_to_num(full_hessian, nan = 0, posinf = 0, neginf = 0 )  # change nan, postive inf , negative inf, to 0
        t_svd = time.time()
        #print('doing EVD')
        # _, eigenvalues, _ = torch.linalg.svd(full_hessian)  # ascending
        #eigenvalues, _  = torch.eig(full_hessian)
        full_hessian = full_hessian.numpy().astype(np.float64)
        full_hessian = (full_hessian + full_hessian.T)/2 # make symetric, to 
        
        #avoid numerical issue
        #full_hessian = full_hessian.cuda()
        #eigenvalues, _  = torch.linalg.eig(full_hessian)
        # eigenvalues, _  = np.linalg.eigh(full_hessian)
        # #_, eigenvalues, _ = np.linalg.svd(full_hessian) 
        # eigenvalues = [eigen.item().real for eigen in eigenvalues]

        # file_name = self.file_dir + 'eigenvalues.txt'
        # with open(file_name, "w") as file:
        #     for item in eigenvalues:
        #         file.write(str(item)+"\n")

        # print(f'EVD time = {time.time()- t_svd}')

        return full_hessian


    def get_full_hessian_layer_by_layer(self):
        self.model.eval()
        self.model.zero_grad(set_to_none = True)

        def hessian_calculation(g_name, g_tensor, batch_idx):
            g_tensor = g_tensor.cuda()
            total_params = g_tensor.size(0)
            hessian_list = []
            t_d = time.time()
            for d in range(total_params):
                unit_vector = torch.zeros(total_params)
                unit_vector[d] = 1
                unit_vector = unit_vector.cuda()
                l = torch.sum(g_tensor*unit_vector)
                l.backward(retain_graph= True)

                hessian_row = []
                for name, param in self.model.named_parameters():
                    if name == g_name:
                        hessian_row.append(param.grad.double().data.clone())
                
                self.model.zero_grad(set_to_none = True)
                hessian_row = [g.flatten() for g in hessian_row] 
                hessian_row = [g.cpu() for g in hessian_row]
                hessian_row = torch.cat(hessian_row)
                hessian_list.append(hessian_row)
                # if d % 1000 == 0:
                #     print(f'Computing hessian: current batch = {batch_idx}/{self.num_batches}, current row of a hessian: {d}/{total_params}, total time = {time.time()- t_d} ')

            hessian = torch.stack(hessian_list, dim = 1)
            return hessian



        full_hessian_dic = {}
        'initialization'
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                #size = torch.flatten(param.data).size(0)
                full_hessian_dic[name] = 0 #torch.zeros(size, size)

        for batch_idx, data in enumerate(self.train_data):

            X_train = data['X_train']
            Y_train = data['Y_train']

            output = self.model(X_train)

            loss = self.loss(output, Y_train) #F.cross_entropy(output, Y_train) 
            loss.backward(create_graph= True)

            g_dic = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_dic[name] = torch.flatten(param.grad.double())

            #g_tensor = torch.cat(g_list, dim = 0)
            self.model.zero_grad(set_to_none = True)

            for name, g_tensor in g_dic.items():
                H = hessian_calculation(name, g_tensor, batch_idx)
                H = torch.nan_to_num(H, nan = 0, posinf = 0, neginf = 0 )  # change nan, postive inf , negative inf, to 0
                H = H.numpy().astype(np.float64)
                H = (H + H.T)/2
                full_hessian_dic[name] = H

        return full_hessian_dic


    def get_batch(self, batch_idx):
        start_idx = batch_idx * self.batch_size * self.block_size
        end_idx = (batch_idx + 1) * self.batch_size * self.block_size
        X = torch.from_numpy((self.train_data[start_idx:end_idx]).astype(np.int64)).reshape(self.batch_size, self.block_size)
        Y = torch.from_numpy((self.train_data[start_idx+1:end_idx+1]).astype(np.int64)).reshape(self.batch_size, self.block_size)
       
        X, Y = X.pin_memory().to(self.device, non_blocking=True), Y.pin_memory().to(self.device, non_blocking=True)

        return X, Y


    def get_true_curve(self, grid, eigenvalues):
        curve = []
        for t in grid:
            density = self.gaussian_density(t, eigenvalues)
            value = np.mean(density)
            curve.append(value)
        return curve
        

    def gaussian_density(self, t, values):
        coeff = 1.0 / np.sqrt(2 * math.pi * self.sigma**2)
        val = -(values - t) ** 2
        val = val / (2.0 * self.sigma**2)
        val = np.exp(val)
        density = coeff * val
        return density

    
    @torch.no_grad()
    def ce_last_layer_hessian_blocks(
        self, 
        input_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        reduction: str = 'mean',  # 'mean' 与 CE 的默认一致；也支持 'sum'
    ):
        """
        仅计算 CE 下最后一层的 C 个 d×d 对角 Hessian block（每类一个）。

        Args:
            input_tensor: X，形状 (B, d)
            prob_tensor: P=softmax(logits)，形状 (B, C)
            reduction: 'mean' 或 'sum'，决定是否除以 B

        Returns:
            一个字典：{ class_index: H_kk (d×d) }
        """
        assert input_tensor.dim() == 2, "input_tensor 必须是 (B,d)"
        assert prob_tensor.dim() == 2, "prob_tensor 必须是 (B,C)"
        B, d = input_tensor.shape
        Bp, C = prob_tensor.shape
        assert B == Bp, "X 与 P 的 batch 大小不一致"

        scale = 1.0 / B if reduction == 'mean' else 1.0

        X = input_tensor  # (B, d)
        blocks: Dict[int, torch.Tensor] = {}

        # 向量化实现：H_kk = scale * X^T (diag(w_k)) X，其中 w_k = p_k(1-p_k)
        for k in range(C):
            p_k = prob_tensor[:, k]  # (B,)
            w_k = p_k * (1.0 - p_k)  # (B,)
            # 等价于 X^T diag(w_k) X ：避免显式diag
            H_kk = (X.t() @ (w_k.unsqueeze(1) * X)) * scale  # (d,d)
            blocks[k] = H_kk

        return blocks


    def calc_classwise_hessian_spectrum(self, 
                                        layer_name="head.weight", 
                                        num_classes=1000):
        """
        计算并保存分类头(lm_head)中不同 class 的 Hessian 谱。
        为了与 get_spectrum_layer_by_layer 的 JSON 结构对齐：
          - classwise_values.json: { "<class>": [[eigvals]*self.num_v] }
          - classwise_weights.json: { "<class>": [[uniform_weights]*self.num_v] }
        其中外层列表长度为 self.num_v，每行长度为 m(=hidden_dim)。
        """
        # ========= 1) 找层 & 逐 batch 聚合每类 Hessian ==========
        for name, param in self.model.named_parameters():
            if name == layer_name:
                W = param
                break
        else:
            raise ValueError(f"Layer {layer_name} not found in model.")

        # m := 每个类块的维度（等于最后一层隐层维度）
        hidden_dim = W.shape[1]
        # 类别数量容错
        if num_classes is None:
            num_classes = W.shape[0]
        num_classes = min(num_classes, W.shape[0])

        # 找到 head 模块（用于 hook 提特征）
        module_name, _ = layer_name.rsplit('.', 1)
        head_module = self.model
        for attr in module_name.split('.'):
            head_module = getattr(head_module, attr)

        num_batches = len(self.train_data)
        class_hessians_sum = {c: np.zeros((hidden_dim, hidden_dim), dtype=np.float64) 
                              for c in range(num_classes)}

        t_hd = time.time()
        print("Calculating Hessian ...")
        for batch_idx, (inputs, targets) in enumerate(self.train_data):
            X = inputs.cuda()
            # Y = targets.cuda()  # CE 的最后一层 Hessian 不需要标签

            # 用 CE 解析式：H_kk = X^T diag(p_k(1-p_k)) X
            with torch.no_grad():
                feat_list = []
                def hook_fn(module, input, output):
                    feat_list.append(input[0])
                handle = head_module.register_forward_hook(hook_fn)
                outputs = self.model(X)
                handle.remove()

                feat = feat_list[0]                          # (B, d)
                prob = torch.softmax(outputs, dim=1)         # (B, C)
                blocks = self.ce_last_layer_hessian_blocks(feat, prob, reduction='mean')
                for c in range(num_classes):
                    H = blocks[c].cpu().numpy().astype(np.float64)
                    class_hessians_sum[c] += H

            if batch_idx % 10 == 1 or batch_idx == self.gradient_accumulation_steps - 1:
                print(f'layer hessian: load_iter={self.ckpt_iteration}, '
                      f'Hd current batch={batch_idx}/{self.num_batches}, '
                      f'time={time.time() - t_hd:.2f}s')
                t_hd = time.time()
            if self.use_minibatch and batch_idx == self.gradient_accumulation_steps - 1:
                break

        # 平均化
        denom = max(1, num_batches)
        class_hessians = {c: class_hessians_sum[c] / denom for c in range(num_classes)}

        # ========= 2) 求特征值，并按经验谱构造权重 ==========
        print("[Hessian] Eigen-decomposition per class ...")
        class_values_json = {}   # { "<c>": [[eigvals]*self.num_v] }
        class_weights_json = {}  # { "<c>": [[uniform]*self.num_v] }
        class_norms_json = {}
        for c in range(num_classes):
            H = class_hessians[c]
            eigvals = np.linalg.eigvalsh(class_hessians[c]).astype(np.float64)
            m = eigvals.shape[0]
            eig_list = eigvals.tolist()
            w_row = (np.ones(m, dtype=np.float64) / max(1, m)).tolist()

            # 关键：外层复制到 num_v 行，确保形状与 Lanczos 一致 (num_v, m)
            values_2d = [list(eig_list) for _ in range(self.num_v)]
            weights_2d = [list(w_row)   for _ in range(self.num_v)]

            class_values_json[str(c)] = values_2d
            class_weights_json[str(c)] = weights_2d
            
            # ======== [ADDED] 计算并记录 Frobenius 与谱范数 ========
            fro_norm = float(np.linalg.norm(H, ord='fro'))                     # [ADDED]
            spectral_norm = float(np.max(np.abs(eigvals)))                     # [ADDED] 对称阵：谱范数=最大奇异值=最大|特征值|
            class_norms_json[str(c)] = {"fro": fro_norm, "spectral": spectral_norm}  # [ADDED]

        # ========= 3) 保存为 JSON ==========
        with open(os.path.join(self.file_dir, 'classwise_values.json'), 'w') as f:
            json.dump(class_values_json, f)
        with open(os.path.join(self.file_dir, 'classwise_weights.json'), 'w') as f:
            json.dump(class_weights_json, f)
        with open(os.path.join(self.file_dir, 'classwise_norms.json'), 'w') as f:   # [ADDED]
            json.dump(class_norms_json, f)                                          # [ADDED]

        print("Saved classwise_values.json and classwise_weights.json with shape (num_v, m) per class.")
        print("Saved classwise_norms.json with Frobenius and spectral norms per class.")  # [ADDED]
        
        
    def plot_classwise_esd(self, select_every: int = 20, out_dirname: str = "classwise_esd"):
        """
        读取 calc_classwise_hessian_spectrum() 生成的 classwise_* JSON，
        对每个 class 估计并绘制 ESD（使用已有的 self.interpolate），
        仅绘制：class 0 以及 (i+1) % select_every == 0 的那些类
        ——即 0、19、39、…（1-based 每 20 个类）。
        """
        # -------- 1) 准备路径与读取 JSON --------
        value_path  = os.path.join(self.file_dir, "classwise_values.json")
        weight_path = os.path.join(self.file_dir, "classwise_weights.json")
        if not os.path.exists(value_path) or not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"缺少 {value_path} 或 {weight_path}。\n"
                "请先运行 calc_classwise_hessian_spectrum() 生成 classwise_* 文件。"
            )

        with open(value_path, "r") as f:
            values_dic = json.load(f)          # { "0": [[...]*num_v], "1": ... }
        with open(weight_path, "r") as f:
            weights_dic = json.load(f)

        # class 索引按数字排序
        class_ids = sorted([int(k) for k in values_dic.keys()])
        if len(class_ids) == 0:
            print("classwise_* 文件为空，跳过绘图。")
            return

        # 仅选择：0 以及 (i+1) % select_every == 0 的类  →  0、19、39、…
        picked = [i for i in class_ids if (i == 0) or ((i + 1) % select_every == 0)]
        print(f"[ESD] 将绘制 {len(picked)}/{len(class_ids)} 个类：", picked)

        # 输出目录
        out_dir = os.path.join(self.file_dir, out_dirname)
        os.makedirs(out_dir, exist_ok=True)

        # -------- 2) 逐类做 ESD & 出图 --------
        for cid in picked:
            key = str(cid)
            # JSON -> ndarray，形状应为 (num_v, m)
            vals = np.asarray(values_dic[key], dtype=np.float64)
            wts  = np.asarray(weights_dic[key], dtype=np.float64)
            
            # 展平后取有限且 >0 的值，去重并降序
            vals_flat = vals.ravel()
            finite_pos = vals_flat[np.isfinite(vals_flat) & (vals_flat > 0)]
            if finite_pos.size > 0:
                uniq_sorted_desc = np.unique(finite_pos)[::-1]
                # 第 5 大（索引 4）；不足 5 个则用最大值（索引 0）
                idx = 4 if uniq_sorted_desc.size >= 5 else 0
                lambda5 = float(uniq_sorted_desc[idx])
            else:
                lambda5 = np.nan

            # 若 lambda5 合法且 >0，则对 vals 做缩放；否则保持原值
            if np.isfinite(lambda5) and (lambda5 > 0):
                scale_note = r" ($\lambda / \lambda_{(5)}$)"
                vals_scaled = vals / lambda5
            else:
                scale_note = ""
                vals_scaled = vals
                if not np.isfinite(lambda5) or lambda5 <= 0:
                    print(f"[WARN] class {cid}: 无有效的第5大特征值，跳过归一化。")

            # 使用你已有的插值核密估计（内部会做 Gaussian 平滑 + 归一化）
            grid, density = self.interpolate(wts, vals)  # density 已归一化为 pdf

            # 线性刻度
            plt.figure(figsize=(6, 4))
            plt.plot(grid, density, lw=1.5, alpha=0.9)
            plt.xlabel("Eigenvalue")
            plt.ylabel("Density")
            plt.title(f"Class {cid} — ESD (linear)")
            plt.tight_layout()
            save_lin = os.path.join(out_dir, f"class_{cid:04d}_esd.png")
            plt.savefig(save_lin, dpi=200)
            plt.close()

            # 对数刻度（y 轴）
            plt.figure(figsize=(6, 4))
            plt.semilogy(grid, density, lw=1.5, alpha=0.9)
            plt.xlabel("Eigenvalue")
            plt.ylabel("Density (log)")
            # 和你其它图风格一致的 y 轴范围，可按需调整
            plt.ylim(max(1e-12, density[np.isfinite(density)].min()*0.5), max(1e-2, density.max()*1.2))
            plt.title(f"Class {cid} — ESD (log-y)")
            plt.tight_layout()
            save_log = os.path.join(out_dir, f"class_{cid:04d}_esd_log.png")
            plt.savefig(save_log, dpi=200)
            plt.close()

            print(f"[ESD] class {cid} saved:\n  {save_lin}\n  {save_log}")
            
            
    @torch.no_grad()
    def ce_last_layer_full_hessian(
        self,
        input_tensor: torch.Tensor,   # X: (B, d)
        prob_tensor: torch.Tensor,    # P_sub: (B, K)，这里 K 可以是整个类，也可以是 subset
        reduction: str = 'mean',
        return_blocks: bool = False
    ) -> torch.Tensor:
        """
        计算交叉熵下最后一层线性分类头在“给定类集合”上的完整 Hessian。
        若 prob_tensor 只包含某个 class subset 的概率列，则结果就是对应子矩阵 H_S。

        Args:
            input_tensor: 特征 X，形状 (B, d)。
            prob_tensor:  对应这些类的概率 P_sub，形状 (B, K)。
            reduction:    'mean' 或 'sum'。
            return_blocks:
                - True  -> 返回块形式 (K, K, d, d)
                - False -> 返回矩阵形式 (K*d, K*d)

        Returns:
            torch.Tensor: (K, K, d, d) 或 (K*d, K*d)，在 input_tensor.device 上。
        """
        assert input_tensor.dim() == 2, "input_tensor 必须是 (B, d)"
        assert prob_tensor.dim() == 2, "prob_tensor 必须是 (B, K)"
        B, d = input_tensor.shape
        Bp, K = prob_tensor.shape
        assert B == Bp, "X 与 P 的 batch 大小不一致"

        scale = 1.0 / B if reduction == 'mean' else 1.0

        X = input_tensor      # (B, d)
        P = prob_tensor       # (B, K)

        # 对 logits 的 Hessian：S_b = diag(p_b) - p_b p_b^T, 形状 (B, K, K)
        diag_p = torch.diag_embed(P)              # (B, K, K)
        outer_p = P.unsqueeze(2) * P.unsqueeze(1) # (B, K, K)
        S = diag_p - outer_p                      # (B, K, K)

        # 特征外积：x_b x_b^T, 形状 (B, d, d)
        x_outer = torch.einsum('bd,be->bde', X, X)  # (B, d, d)

        # 聚合：H_{kl} = sum_b S[b,k,l] * x_b x_b^T
        H_blocks = torch.einsum('bkl,bij->klij', S, x_outer) * scale  # (K, K, d, d)

        if return_blocks:
            return H_blocks

        H_full = H_blocks.reshape(K * d, K * d)
        # print("H_full", H_full)
        return H_full

    
    def calc_last_layer_subset_hessian_heatmap(
        self,
        layer_name: str = "head.weight",
        class_indices: Optional[np.ndarray] = None,
        reduction: str = "mean",
        save_numpy: bool = True,
        save_heatmap: bool = True
    ):
        """
        只对指定 class subset（例如 top-k frequent classes）计算
        last layer 分类头的完整 Hessian 子矩阵，并画热图。

        - logits = head(feat) = feat @ W^T, W.shape = (C_total, d)
        - subset 大小为 K => Hessian 子矩阵形状为 (K*d, K*d)

        Args:
            layer_name:    最后一层权重参数名，例如 "head.weight"。
            class_indices: 一个 1D numpy 数组或 list，指定要保留的类别 ID（相对于 W 的行索引）。
            reduction:     'mean' 或 'sum'。
            save_numpy:    是否保存 .npy 文件。
            save_heatmap:  是否保存热图 .png。
            show:          是否 plt.show()（本地可视化用）。
        """
        # ========= 0) 找到 head.weight ==========
        for name, param in self.model.named_parameters():
            if name == layer_name:
                W = param
                break
        else:
            raise ValueError(f"Layer {layer_name} not found in model.")

        total_classes = W.shape[0]
        hidden_dim = W.shape[1]

        # ========= 1) 决定 class subset ==========
        if class_indices is None:
            raise ValueError("必须指定 class_indices")
        else:
            class_indices = np.asarray(class_indices, dtype=np.int64)

        # 安全检查
        assert class_indices.ndim == 1
        assert np.all(class_indices >= 0) and np.all(class_indices < total_classes)
        # 统一排序（可选）
        class_indices = np.unique(class_indices)
        K = class_indices.shape[0]
        print(f"[Hessian] Using class subset of size K={K}: {class_indices.tolist()}")

        # 存一下 subset 映射信息，方便之后对齐
        subset_info_path = os.path.join(self.file_dir, "last_layer_subset_classes.json")
        with open(subset_info_path, "w") as f:
            json.dump({"class_indices": class_indices.tolist()}, f)
        print(f"[Hessian] Saved subset class indices to {subset_info_path}")

        # ========= 2) 找到 head 模块（用于 hook 提特征） ==========
        module_name, _ = layer_name.rsplit('.', 1)
        head_module = self.model
        for attr in module_name.split('.'):
            head_module = getattr(head_module, attr)

        num_batches = len(self.train_data)

        # Hessian 子矩阵形状：(K*d, K*d)
        Hd = K * hidden_dim
        H_sum = np.zeros((Hd, Hd), dtype=np.float64)

        t_hd = time.time()
        print("Calculating subset full last-layer Hessian ...")

        class_indices_t = torch.as_tensor(class_indices, dtype=torch.long, device="cuda")

        for batch_idx, (inputs, targets) in enumerate(self.train_data):
            X = inputs.cuda(non_blocking=True)

            with torch.no_grad():
                # --- hook: 抓 head 输入特征 ---
                feat_list = []

                def hook_fn(module, input, output):
                    feat_list.append(input[0])  # (B, d)

                handle = head_module.register_forward_hook(hook_fn)
                outputs = self.model(X)           # logits: (B, C_total)
                handle.remove()

                feat = feat_list[0]               # (B, d)
                prob_full = torch.softmax(outputs, dim=1)  # (B, C_total)

                # 只取 subset 类的概率列 (B, K)
                prob_sub = prob_full.index_select(dim=1, index=class_indices_t)

                # --- 当前 batch 在 subset 上的完整 Hessian ---
                H_batch = self.ce_last_layer_full_hessian(
                    feat,
                    prob_sub,
                    reduction=reduction,
                    return_blocks=False,   # (K*d, K*d)
                )
                H_sum += H_batch.cpu().numpy().astype(np.float64)

            # --- 日志 ---
            if batch_idx % 10 == 1 or batch_idx == self.gradient_accumulation_steps - 1:
                print(
                    f'subset head hessian: iter={self.ckpt_iteration}, '
                    f'batch={batch_idx}/{num_batches}, '
                    f'time={time.time() - t_hd:.2f}s'
                )
                t_hd = time.time()

            if getattr(self, "use_minibatch", False) and batch_idx == self.gradient_accumulation_steps - 1:
                break

        # ========= 3) 平均化 ==========
        denom = max(1, num_batches)
        H = H_sum / denom   # (K*d, K*d)
        
        # ========= 3.1) 计算 condition number & modified condition number ==========
        cond, modified_cond = self.compute_condition_numbers_from_hessian(H)
        
        # ========= 3.2) 计算 Hessian 主对角能量 ratio =========  # NEW
        abs_diag_sum = np.sum(np.abs(np.diag(H)))          # 主对角绝对值之和  # NEW
        abs_total_sum = np.sum(np.abs(H))                  # 整个 Hessian 绝对值之和  # NEW
        if abs_total_sum == 0:                             # 避免除零  # NEW
            diag_energy_ratio = 0.0                        # NEW
        else:                                              # NEW
            diag_energy_ratio = abs_diag_sum / abs_total_sum   # NEW

        txt_path = os.path.join(
            self.file_dir,
            f"last_layer_subset_hessian_condition_{layer_name.replace('.', '_')}.txt"
        )
        with open(txt_path, "w") as f:
            f.write(f"Hessian subset for layer: {layer_name}\n")
            f.write(f"Subset classes (K={K}): {class_indices.tolist()}\n")
            f.write(f"Hessian shape: {H.shape[0]} x {H.shape[1]}\n")
            f.write(f"Condition number: {cond:.6e}\n")
            f.write(
                "Modified condition number "
                "(sigma_max / mean of smallest 10% singular values): "
                f"{modified_cond:.6e}\n"
            )
            f.write(                                           # NEW
                "Hessian diagonal energy ratio "               # NEW
                "(sum |diag(H)| / sum |H|): "                  # NEW
                f"{diag_energy_ratio:.6e}\n"                   # NEW
            )

        # ========= 4) 保存 npy ==========
        if save_numpy:
            npy_path = os.path.join(self.file_dir, "last_layer_subset_hessian.npy")
            np.save(npy_path, H)
            print(f"[Hessian] Saved subset Hessian to {npy_path}, shape={H.shape}")

        # ========= 5) 画热图 ==========
        if save_heatmap:
            dist_table = np.abs(H)

            H_MAX = np.max(dist_table)
            if H_MAX == 0:
                H_MAX = 1e-12

            # 如果你想只画下三角，可以这样构造 mask（可选）：
            # mask = np.triu(np.ones_like(dist_table, dtype=bool), k=1)
            # 如果想看完整矩阵，就设为 None：
            mask = None

            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(
                dist_table,
                cmap="coolwarm",
                mask=mask,
                square=True,
                cbar_kws={"label": "Hessian Entry Value"},
                vmin=0.0,
                vmax=0.002,
                xticklabels=False,
                yticklabels=False,
            )
            plt.title(
                f"Abs Hessian Heatmap (subset, layer={layer_name}, K={K}, d={hidden_dim})"
            )
            save_path = os.path.join(
                self.file_dir,
                f"subset_hessian_heatmap_{layer_name.replace('.', '_')}.png"
            )
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"[Hessian] Saved subset heatmap to {save_path}")
            
            
    def compute_condition_numbers_from_hessian(
        self, 
        H: np.ndarray,
        eps: float = 1e-12,
    ):
        """
        给定对称 Hessian H (n×n)，计算：
            - cond(H) = σ_max / σ_min
            - modified_cond(H) = σ_max / mean(后 10% 奇异值)

        对称阵：奇异值 = |特征值|，所以用 eigvalsh 即可。
        eps 用来过滤数值噪声非常小的奇异值。
        """
        assert H.ndim == 2 and H.shape[0] == H.shape[1], "H 必须是方阵"

        # 1) 特征值 -> 奇异值 = |λ|
        eigvals = np.linalg.eigvalsh(H).astype(np.float64)
        sing_vals = np.abs(eigvals)

        # 2) 过滤掉数值上接近 0 的奇异值，避免 0 做分母
        valid = np.sort(sing_vals[sing_vals > eps])   # 升序
        if valid.size == 0:
            # 全是 0 或非常小，条件数视为 +inf
            return float("inf"), float("inf")

        sigma_max = float(valid[-1])
        sigma_min = float(valid[0])

        # --- 标准 condition number ---
        cond = float("inf") if sigma_min < eps else float(sigma_max / sigma_min)

        # --- modified condition number ---
        # “后 10%”：这里取的是“最小的 10% 奇异值”的平均值，
        # 也就是有效奇异值数组 valid 的前 10%。
        k = max(1, valid.size // 10)   # 至少取 1 个
        tail = valid[:k]               # 最小的 10%
        tail_mean = float(np.mean(tail))

        if tail_mean < eps:
            modified_cond = float("inf")
        else:
            modified_cond = float(sigma_max / tail_mean)

        return cond, modified_cond
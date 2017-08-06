import numpy as np


class Fixations:
    def __init__(self, net):
        self.net = net

    def fc(self, points, layer, prevLayer):
        for cr in range(5):
            if (points[cr] != 0):
                layer_out = []
                # Blob values for prevous layer
                data = np.squeeze(self.net.blobs[prevLayer].data[cr, :])
                # Weights for the current layer
                param = self.net.params[layer][0].data
                # If previous layer is not fully connected
                if (data.ndim == 3):
                    shape = data.shape
                    data = np.reshape(data, [param.shape[1], ])
                    conv = data*param
                    for i in points[cr]:
                        # Getting Top positions for each input
                        position = np.argmax(conv[i, :])
                        layer_out.append(np.unravel_index(position, shape))
                else:
                    conv = data*param
                    for i in points[cr]:
                        # Getting Top-num activations for each input
                        num = np.sum(conv[i, :] > 0)
                        layer_out.extend(np.argsort(conv[i, :])[::-1][:num])
                points[cr] = list(set(layer_out))
        return points

    def pool(self, points, prevLayer, K, S):
        for cr in range(5):
            if (points[cr] != 0):
                layer_out = []
                for i in points[cr]:
                    if not isinstance(i, tuple):
                        i = [i] + [0, 0]
                    # Getting the receptive region the pool operates on
                    x = (S*i[1], S*i[1] + K)
                    y = (S*i[2], S*i[2] + K)
                    # Getting most contributing x and y relative to the
                    # region being operated on
                    blob = self.net.blobs[prevLayer].data[cr, i[0], x[0]:x[1], y[0]:y[1]]
                    x1, y1 = np.unravel_index(np.argmax(blob), blob.shape)
                    layer_out.append((i[0], x[0]+x1, y[0]+y1))
                points[cr] = list(set(layer_out))
        return points

    def conv(self, points, layer, prevLayer, K, S, P, group=False):
        # Only supports group = 2
        for cr in range(5):
            if (points[cr] != 0):
                layer_out = []
                for i in points[cr]:
                    x = (S*i[1], S*i[1]+K)
                    y = (S*i[2], S*i[2]+K)
                    flag = False  # To caluculate feature for grouping
                    if group:
                        p_s = self.net.params[layer][0].data.shape[1]
                        b_s = self.net.blobs[prevLayer].data.shape[1]
                        if (i[0] >= p_s/2):  # Choosing the group
                            flag = True
                            if (P != 0):
                                data = np.lib.pad(self.net.blobs[prevLayer].data[
                                    cr, b_s/2:b_s, :, :], P, 'constant',
                                    constant_values=0)[P:-P, x[0]:x[1], y[0]:y[1]]
                            else:
                                data = self.net.blobs[prevLayer].data[cr, b_s/2:b_s, x[0]:x[1], y[0]:y[1]]
                        else:
                            if (P != 0):
                                data = np.lib.pad(self.net.blobs[prevLayer].data[
                                    cr, :b_s/2, :, :], P, 'constant',
                                    constant_values=0)[P:-P, x[0]:x[1], y[0]:y[1]]
                            else:
                                data = self.net.blobs[prevLayer].data[cr, :b_s/2, x[0]:x[1], y[0]:y[1]]
                    else:
                        if (P != 0):
                            data = np.lib.pad(self.net.blobs[prevLayer].data[
                                    cr, :, :, :], P, 'constant',
                                    constant_values=0)[P:-P, x[0]:x[1], y[0]:y[1]]
                        else:
                                data = self.net.blobs[prevLayer].data[cr, :, x[0]:x[1], y[0]:y[1]]
                    param = self.net.params[layer][0].data[i[0], :, :, :]
                    conv = data*param
                    # Getting most contributing position
                    feature = np.argmax(np.sum(np.sum(conv, axis=2), axis=1))
                    if (flag):
                        feature += b_s/2
                    layer_out.append((feature, x[0], y[0]))
                points[cr] = list(set(layer_out))
        return points

    def data(self, points, inc, resFac):
        output = []
        for cr in range(5):
            if (points[cr] != 0):
                layer_out = []
                # Bringing points back to image size
                for i in points[cr]:
                    layer_out.append((int((i[1]+inc[cr][0])*(1/resFac)),
                                      int((i[2]+inc[cr][1])*(1/resFac))))
                output.extend(layer_out)
        return output
    
    # Inception layer for GoogLeNet
    def inception(self, points, layer, prevLayer, pool=False, out=False, prevLayer2=None):
        for cr in range(5):
            if (points[cr] != 0):
                layer_out = []
                for i in points[cr]:
                    # Just to handle naming
                    if pool:
                        add = ''
                    else:
                        add = '/output'
                    # Getting the ranges of each branch in the inception layer
                    num_f1 = self.net.blobs[layer+'/1x1'].data.shape[1]
                    num_f2 = self.net.blobs[layer+'/3x3'].data.shape[1]
                    num_f3 = self.net.blobs[layer+'/5x5'].data.shape[1]
                    # Checking if we need to take the 1x1 conv path
                    if i[0] < num_f1:
                        param = self.net.params[layer+'/1x1'][0].data[i[0], :, 0, 0]
                        data = self.net.blobs[prevLayer+add].data[cr,:, i[1], i[2]]
                        conv = np.argmax(data*param)
                        values = (conv, i[1], i[2])
                    # Checking if we need to take the 3x3 conv path
                    elif i[0] < num_f1 + num_f2:
                        param = self.net.params[layer+'/3x3'][0].data[i[0]-num_f1,:,:,:]
                        data = np.lib.pad(self.net.blobs[layer+'/3x3_reduce'].data[cr,:,:,:], 1, 'constant'
                                          ,constant_values=0)[1:-1, i[1]:i[1]+3, i[2]:i[2]+3]
                        feature = np.argmax(np.sum(np.sum(data*param, axis=2), axis=1))
                        param = self.net.params[layer+'/3x3_reduce'][0].data[feature, :, 0, 0]
                        data = self.net.blobs[prevLayer+add].data[cr, :, i[1], i[2]]
                        conv = np.argmax(data*param)
                        values = (conv,i[1],i[2])
                    # Checking if we need to take the 5x5 conv path
                    elif i[0] < num_f1 + num_f2 + num_f3:
                        param = self.net.params[layer+'/5x5'][0].data[i[0]-num_f1-num_f2, :, :, :]
                        data =  np.lib.pad(self.net.blobs[layer+'/5x5_reduce'].data[cr,:,:,:], 2, 'constant',
                                           constant_values=0)[2:-2,i[1]:i[1]+5,i[2]:i[2]+5]
                        feature = np.argmax(np.sum(np.sum(data*param,axis=2),axis=1))
                        param = self.net.params[layer+'/5x5_reduce'][0].data[feature,:,0,0]
                        data = self.net.blobs[prevLayer+add].data[cr, :, i[1], i[2]]
                        conv = np.argmax(data*param)
                        values = (conv, i[1], i[2])
                    # Otherwise taking the pool path
                    else:
                        param = self.net.params[layer+'/pool_proj'][0].data[i[0]-num_f1-num_f2-num_f3, :, 0, 0]
                        data = self.net.blobs[layer+'/pool'].data[cr, :, i[1], i[2]]
                        feature = np.argmax(data*param)
                        data = np.lib.pad(self.net.blobs[prevLayer+add].data[cr, :, :, :],1,'constant',
                                          constant_values=0)[feature+1, i[1]:i[1]+3, i[2]:i[2]+3]
                        x1, y1 = np.unravel_index(np.argmax(data), data.shape)
                        values = (feature, i[1]+x1-1, i[2]+y1-1)
                    layer_out.append(values)
                points[cr] = list(set(layer_out))
        return points
    
    # Residual block for ResNet-101
    def res(self, points, layer, prevLayer):
        for cr in range(5):
            if (points[cr] != 0):
                layer_out = []
            flag = False #flag to check if previous block downsampled the inputs
            if 'a' in layer:
                    flag = True
                    # Getting activations from previous residual block
                    branch_skip = self.net.blobs['res'+layer+'_branch1'].data[cr]
            else:
                    branch_skip = self.net.blobs['res'+prevLayer].data[cr]
            # Getting activations from all convolution layers inside the residual block
            branch_res_blobs = [self.net.blobs['res'+layer+'_branch2a'].data[cr], self.net.blobs['res'+layer+'_branch2b'].data[cr], self.net.blobs['res'+layer+'_branch2c'].data[cr]]
        # Pad input to 3x3 block so the size remains same after conv
            branch_res_blobs[0] = np.lib.pad(branch_res_blobs[0], 1 ,'constant',constant_values=0)[1:-1,:,:]
            for i in points[cr]:
                # To check if the higher activation at a point came from skip or delta branch
                if branch_skip[i[0], i[1], i[2]] >= branch_res_blobs[2][i[0], i[1], i[2]]:
                        if flag:
                                param = self.net.params['res'+layer+'_branch1'][0].data[i[0], :, 0, 0]
                                if prevLayer == 'pool1':
                                        data = self.net.blobs[prevLayer].data[cr, :, i[1], i[2]]
                                        feature = np.argmax(data*param)
                                        layer_out.append((feature, i[1], i[2]))
                                else:
                                        data = self.net.blobs['res'+prevLayer].data[cr, :, i[1]*2, i[2]*2]
                                        feature = np.argmax(data*param)
                                        layer_out.append((feature, i[1]*2, i[2]*2))
                        else:
                                layer_out.append(i)
                else:
                        param = self.net.params['res'+layer+'_branch2c'][0].data[i[0], :, 0, 0]
                        feature = np.argmax(branch_res_blobs[1][:, i[1], i[2]]*param)
                        param = self.net.params['res'+layer+'_branch2b'][0].data[feature, :, :, :]
                        data = branch_res_blobs[0][:, i[1]:i[1]+3, i[2]:i[2]+3]
                        feature = np.argmax(np.sum(np.sum(data*param, axis=2),axis=1))
                        x, y = np.unravel_index(np.argmax(data[feature,:,:]),data[feature,:,:].shape)
                        if flag:
                                param = self.net.params['res'+layer+'_branch2a'][0].data[feature, :, 0, 0]
                                if prevLayer =='pool1':
                                        feature = np.argmax(self.net.blobs[prevLayer].data[cr, :, i[1]+x-1, i[2]+y-1]*param)
                                        layer_out.append((feature, i[1]+x-1, i[2]+y-1))
                                else:
                                        feature = np.argmax(self.net.blobs['res'+prevLayer].data[cr, :, (i[1]+x-1)*2, (i[2]+y-1)*2]*param)
                                        layer_out.append((feature, (i[1]+x-1)*2, (i[2]+y-1)*2))
                        else:
                                param = self.net.params['res'+layer+'_branch2a'][0].data[feature, :, 0, 0]
                                feature = np.argmax(branch_skip[:,i[1]+x-1,i[2]+y-1]*param)
                                layer_out.append((feature, i[1]+x-1, i[2]+y-1))
            points[cr] = list(set(layer_out))                        
        return points

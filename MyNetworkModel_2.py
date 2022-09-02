import numpy as np
import time
import networkx as nx
class Networks:
    def __init__(self,G):
        self.G = G
        self.totNodes = len(list(G.nodes))
        self.node_list = list(set(G.nodes))
        self.Sample_list = []
        self.Sample_Index = 0
        self.Sample_prob_val= dict()
    def __total_nodes__(self):
        return self.totNodes
    
    def vec_Forman_edge_curvature_1(self,node_i,node_j):
    #incident_sum of node i
        wt_ij = self.G[node_i][node_j]['weight']
        incident_sum_11 = 0
        for nbr_k in self.G[node_i]:
            if nbr_k != node_j:
                wt_k = self.G[node_i][nbr_k]['weight']
                if  wt_ij*wt_k != 0:
                    incident_sum_11 += self.G.degree(node_i)/np.sqrt((wt_ij*wt_k))  
        #incident sum of node_j
        incident_sum_12 = 0
        for nbr_k in self.G[node_j]:
            if nbr_k != node_i:
                wt_k = self.G[node_j][nbr_k]['weight']
                if  wt_ij*wt_k != 0:
                    incident_sum_12 += self.G.degree(node_j)/np.sqrt((wt_ij*wt_k))  
        return wt_ij*((self.G.degree(node_i)+self.G.degree(node_j))/(wt_ij+0.000001) - incident_sum_11-incident_sum_12 )

    def vec_Forman_node_curvature_1(self,node_i):
        tmp = [self.vec_Forman_edge_curvature_1(node_i,node_j) for node_j in list(self.G[node_i].keys())]
        return sum(tmp)

    def vec_Forman_abs_node_curvature_1(self,node_i):
        tmp = [np.abs(self.vec_Forman_edge_curvature_1(node_i,node_j)) for node_j in list(self.G[node_i].keys())]
        return sum(tmp)
    

    def vec_rfc(self,node_i,node_j):
        # relative_forman_curvature
        unscaled_p = np.abs(self.vec_Forman_edge_curvature_1(node_i,node_j))
        return unscaled_p

    def vec_rfc_dd(self,node_i,node_j):
        # relative_forman_curvature_Degree_Divided
        unscaled_p = np.abs(self.vec_Forman_edge_curvature_1(node_i,node_j))/self.G.degree(node_i)
        return unscaled_p
    
    def vec_rfc_cc(self,node_i,node_j,alpha_j):
        # relative_fc_with_cc
        unscaled_p = alpha_j *np.abs(self.vec_Forman_edge_curvature_1(node_i,node_j))
        return unscaled_p
    
    def vec_rfc_cc_dd(self,node_i,node_j,alpha_j):
        # relative_fc_with_cc_Degree_Divided/Haantjes/Assortativity Coefficient
        unscaled_p = alpha_j *np.abs(self.vec_Forman_edge_curvature_1(node_i,node_j))/self.G.degree(node_i)
        return unscaled_p
    def vec_rfc_comms(self,node_i,node_j,alpha_j):
        # relative_forman_curvature_comms
        # unscaled_p = alpha_j *np.abs(self.vec_Forman_edge_curvature_1(node_i,node_j))/self.G.degree(node_i)
        unscaled_p = alpha_j *np.abs(self.vec_Forman_edge_curvature_1(node_i,node_j)) + (1-alpha_j)*(self.vec_Forman_abs_node_curvature_1(node_j)- self.vec_Forman_edge_curvature_1(node_i,node_j))
        return unscaled_p
    
    def vec_rfc_comms_dd(self,node_i,node_j,alpha_j):
        # relative_forman_curvature_comms
        # unscaled_p = alpha_j *np.abs(self.vec_Forman_edge_curvature_1(node_i,node_j))/self.G.degree(node_i)
        unscaled_p = alpha_j *np.abs(self.vec_Forman_edge_curvature_1(node_i,node_j))/self.G.degree(node_i) + (1-alpha_j)*(self.vec_Forman_abs_node_curvature_1(node_j)- self.vec_Forman_edge_curvature_1(node_i,node_j))/self.G.degree(node_j)
        return unscaled_p
    
    def vec_uniform(self,node_i,Node_j):
        return 1

    def random_node_generator(self,size = 1,replace = True):
        return np.random.choice(self.node_list, size = size, replace=replace)

    def vec_scale(self,vec):
        if type(vec).__module__ == np.__name__ :
            tt = vec.sum()
            return vec/tt
        else:
            tt = sum(vec)
            return [i/tt for i in vec]
            
    def to_vec_prob(self, node_i, vec_func, alpha_j = None):
        if not(alpha_j is None):
            tmp = np.array([vec_func(node_i,node_j,alpha_j) for node_j in list(self.G[node_i].keys())])
            return self.vec_scale(tmp)
        else:
            tmp = [vec_func(node_i,node_j) for node_j in list(self.G[node_i].keys())]
            return self.vec_scale(tmp)
    
        
    # def MCMC_Sample_generator(self, Initial_state, vec_func, proportion = 0.5, iters = None):
    # # for specific iteration
    #     proportion_of_nodes = np.floor(proportion * self.totNodes) # Size of subgraph
        
    #     Current_state = Initial_state
    #     self.Sample_list.append([Current_state])
        
    #     iteration = 0
    #     while (iteration<= proportion_of_nodes) or (len(self.Sample_list[self.Sample_Index]) <= proportion_of_nodes):
    #         probs = self.to_vec_prob(node_i=Current_state,vec_func = vec_func)
    #         change = np.random.choice( self.G[Current_state], replace= False, p =probs)
    #         self.Sample_list[self.Sample_Index].append(change)
    #         self.Sample_list[self.Sample_Index] = list(set(self.Sample_list[self.Sample_Index]))
    #         Current_state = change
    #         iteration += 1

            
    def MCMC_Sample_generator(self, Initial_state, vec_func = None ,alpha_j = None, proportion = 0.5, iters = False):
        if vec_func is None:
            vec_func = self.vec_uniform
            #Default values is rfc_dd
        proportion_of_nodes = np.floor(proportion * self.totNodes) # Size of subgraph
        print(f'{self.Sample_Index+1}th Session Start....')
        t_start = time.perf_counter()
        Current_state = Initial_state
        self.Sample_list.append([Current_state])
        self.Sample_prob_val[f'{vec_func.__name__}'] = {Current_state:None}
        # iteration = 0
        if iters:
            while (len(self.Sample_list[self.Sample_Index]) <= proportion_of_nodes):
                
                #check whether saved or not
                if self.Sample_prob_val[f'{vec_func.__name__}'].get(Current_state) is None:
                    probs = self.to_vec_prob(node_i=Current_state,vec_func = vec_func,alpha_j = alpha_j)
                    self.Sample_prob_val[f'{vec_func.__name__}'][Current_state] = probs
                probs =  self.Sample_prob_val[f'{vec_func.__name__}'][Current_state]
                change = np.random.choice( self.G[Current_state], replace= False, p = probs)
                self.Sample_list[self.Sample_Index].append(change)
                self.Sample_list[self.Sample_Index] = list(set(self.Sample_list[self.Sample_Index]))
                Current_state = change
            
        else:
            for _ in range(int(proportion_of_nodes)) :
                if self.Sample_prob_val[f'{vec_func.__name__}'].get(Current_state)  is None:
                    probs = self.to_vec_prob(node_i=Current_state,vec_func = vec_func,alpha_j = alpha_j)
                    self.Sample_prob_val[f'{vec_func.__name__}'][Current_state] = probs
                probs =  self.Sample_prob_val[f'{vec_func.__name__}'][Current_state]
                change = np.random.choice( self.G[Current_state], replace= False, p =probs)
                self.Sample_list[self.Sample_Index].append(change)
                self.Sample_list[self.Sample_Index] = list(set(self.Sample_list[self.Sample_Index]))
                Current_state = change
        t_end  = time.perf_counter()
        print(f'Session end...{t_end-t_start}')
        self.Sample_Index +=1
        return self.Sample_list[-1]

    def sub_graph(self, nodes):
        #nodes of the subgraph
        sub_G = self.G.subgraph(nodes)
        return sub_G

    def closeness_criteria(self, sub_G):
        # return values
        GS = nx.closeness_centrality(self.G)
        GS_mean = np.mean(list(GS.values()))
        
        GS_sub = nx.closeness_centrality(sub_G)
        GS_sub_mean = np.mean(list(GS_sub.values()))
        return (GS_mean - GS_sub_mean)**2

    def betweenness_criteria(self,sub_G):
        # return values
        GS = nx.betweenness_centrality(self.G, weight = 'weight')
        GS_mean = np.mean(list(GS.values()))
        
        GS_sub = nx.betweenness_centrality(sub_G, weight = 'weight')
        GS_sub_mean = np.mean(list(GS_sub.values()))
        return (GS_mean - GS_sub_mean)**2
    
    def betweenness_dif_percent(self,sub_G):
        # return value
        GS = nx.betweenness_centrality(self.G, weight = 'weight')
        GS_mean = np.mean(list(GS.values()))
        
        GS_sub = nx.betweenness_centrality(sub_G, weight = 'weight')
        GS_sub_mean = np.mean(list(GS_sub.values()))
        return abs(GS_mean - GS_sub_mean)/GS_mean *100

    def closeness_dif_percent(self, sub_G):
        # return values
        GS = nx.closeness_centrality(self.G)
        GS_mean = np.mean(list(GS.values()))
        
        GS_sub = nx.closeness_centrality(sub_G)
        GS_sub_mean = np.mean(list(GS_sub.values()))
        return abs(GS_mean - GS_sub_mean)/GS_mean *100

    def Best_sampler(self, samples = None,vec_fun = None, alpha_fac = None, N=100):
        
        if samples is None:
            rd_inits = self.random_node_generator(N)
            samples = [self.MCMC_Sample_generator(Initial_state=state, proportion=0.8, vec_func= vec_fun) for state in rd_inits]
        N = len(samples)
        sample_node_ls = []
        for val in samples:
            sample_node_ls+=val
        sample_node_ls = list(set(sample_node_ls))
        node_occurace = {val:len([1 for sample in samples if val in sample]) for val in sample_node_ls}
        alpha = min([len(val)/self.totNodes  for val in samples])
        if not alpha_fac is None:
            alpha *= alpha_fac
        best_sample = [val for val in node_occurace.keys() if node_occurace[val] > alpha*N]
        return best_sample
  

    
    

# Standard Libraries
import time
import argparse
import os
import warnings
from tqdm import tqdm  # For progress bars
import pickle  # For object serialization

# Data Science and Machine Learning Libraries
import numpy as np
import pandas as pd  # For data manipulation

# PyTorch Libraries
import torch  # Main PyTorch import
# Plotting Libraries
import matplotlib.pyplot as plt  # For visualization

# Custom module imports
from src.LNDA import *
from src.Preprocess import Preprocess
from src.SKANTHP import SKANTHP

def final_train(L,Omega,x,X,A_emb,W_emb,num_vertices,num_types,Sequences,N, D, alpha, Beta, gamma, Phi,rho,A,mue,F,phi,N_EPOCH = 1000,model_type=None,LNDA=0,TOL = 0.1,verbose = True,lb = -np.inf):   
    """Main training function for the model using Variational EM algorithm"""
    LAMBDAS=Omega
    for epoch in range(N_EPOCH): 
        # Variational EM Algorithm
        # E-step: Update variational parameters
        if model_type==None:
            if LNDA:
                # Nested version of E-step
                Phi, gamma,rho = E_step_nested(Omega,D,F,N,X,Phi, gamma, alpha, Beta,rho,A,mue,x,phi,L,K=2)
            else:
                Phi, gamma,rho = E_step(Omega,D,F,N,X,Phi, gamma, alpha, Beta,rho,A,mue,x,phi,L)
        else:
            LAMBDAS = []
            for i in range(D):
                # Collected Hawkes intensity from
                LAMBDAS.append(SKANTHP(mue[i], A_emb, W_emb, num_vertices, num_types, Sequences[i], model_type=model_type))
            if LNDA:
                # Enhanced nested E-step
                Phi, gamma, rho = E_step_nested_enhanced(Omega, D, F, N, X, Phi, gamma, alpha, rho, A, phi, L, LAMBDAS, K=2)
            else:
                Phi, gamma, rho = E_step_enhanced(Omega, D, F, N, X, Phi, gamma, alpha, rho, A, phi, L, LAMBDAS)
        
        # M-step: Update model parameters
        if LNDA:
            alpha, Beta, mue = M_step_nested(F,gamma, alpha, Beta, D, N, x,mue,F)
        else:
            alpha, Beta, mue = M_step(gamma, alpha, Beta, D, N, x,mue,F)
    return Phi, gamma,rho, alpha, Beta, mue,LAMBDAS
class Results:
    """Container class for storing model results and parameters"""

    def __init__(self, params):
        self.Phi, self.gamma, self.rho, self.alpha, self.Beta, self.mue,self.Omega = params
        self.time=0 #training time
    def forward(self):
        pass       
# Metric calculation functions
def MAE(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-13))) * 100  # Added epsilon to avoid division by zero

def RMSE(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def MSE(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def FECA(y_true, y_pred):
    """Fraction of Energy Consumption Accurate"""
    return 1 - (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))

def all_metrics(y_true, y_pred):
    """Calculate all evaluation metrics and return as DataFrame"""
    metrics = {
        'MAE': MAE(y_true, y_pred),
        'MAPE': MAPE(y_true, y_pred),
        'RMSE': RMSE(y_true, y_pred),
        'MSE': MSE(y_true, y_pred),
        'FECA': FECA(y_true, y_pred)
    }
    return pd.DataFrame([metrics])
def main():
    """Main execution function handling command-line arguments, training, and evaluation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='REDD')  # Dataset selection
    parser.add_argument('-num_types', type=int, default=3)  # Number of appliance types
    parser.add_argument('-iteration', type=int,default="")  # Experiment iteration number
    parsed = parser.parse_args()

    # Model configuration
    LDA_LNDA = ["LDA", "LNDA"]  # Model variants
    All_Models = ["HP", "STHP", "SKANTHP"]  # Model types
    models = [None, 'mlp', 'kan_original']  # Model architectures
    All_Results = {}  # Store all model results

    # Data preprocessing
    Kmeans_mods, W_emb, A_emb, x, testings, y_trains, X, All_sequences, num_vertices, allhouses, Appliances, inputs, Dataset, Len_of_inputs, docs, vocab, num_types = Preprocess(parsed.data, parsed.num_types, 2)

    # Model training loop
    for ii, LL in enumerate(LDA_LNDA):
        for mod, model in zip(All_Models, models):
            start_time = time.time()
            # Initialize model parameters
            L, Omega, N, D, alpha, Beta, gamma, Phi, rho, A, mue, F, phi = init_lda(docs, vocab, testings, y_trains, x)
            # Train model
            All_Results[LL + '_' + mod] = Results(
                final_train(L, Omega, x, X, A_emb, W_emb, num_vertices, num_types, All_sequences, N, D, alpha, Beta, gamma, Phi, rho, A, mue, F, phi, 1, model, ii)
            )
            end_time = time.time()
            All_Results[LL + '_' + mod].time = end_time - start_time

    # Results processing and visualization
    all_metrics_with_houses = []
    all_houses = []
    wind = 100  # Window size for visualization
    os.makedirs(parsed.data, exist_ok=True)  # Create output directory

    # Per-house evaluation
    for m in range(allhouses):
        metrics_list = []  # Store metrics per appliance
        house = []

        # Create figure objects for visualizations
        ax=plt.figure(figsize=[int(np.ceil(num_vertices/2))*8, 2*4])
        ax_1ap=plt.figure(figsize=[15, 5])
        ax_LDALNDA=plt.figure(figsize=[int(np.ceil(num_vertices/2))*8, 2*4])
        ax_LDASTHPLNDASTHP=plt.figure(figsize=[int(np.ceil(num_vertices/2))*8, 2*4])
        ax_LDASKANTHPLNDASKANTHP=plt.figure(figsize=[int(np.ceil(num_vertices/2))*8, 2*4])
        ax_LNDASTHPLNDASKANTHP=plt.figure(figsize=[int(np.ceil(num_vertices/2))*8, 2*4])
        ax_LDASTHPLDASKANTHP=plt.figure(figsize=[int(np.ceil(num_vertices/2))*8, 2*4])
        ax_LDALDASTHP=plt.figure(figsize=[int(np.ceil(num_vertices/2))*8, 2*4])
        ax_LNDALNDASTHP=plt.figure(figsize=[int(np.ceil(num_vertices/2))*8, 2*4])
        
        # Set figure titles
        ax_1ap.suptitle(f'{Dataset} - House {m}, LDA_HP vs LNDA_SKANTHP', y=1.02, fontsize=24)
        ax.suptitle(f'{Dataset} LNDA: House {m}', y=1.02, fontsize=24)
        ax_LDALNDA.suptitle(f'{Dataset} - House {m}, LDA_HP vs LNDAhp', y=1.02, fontsize=24)
        ax_LDASTHPLNDASTHP.suptitle(f'{Dataset} - House {m}, LDA_STHP vs LNDA_STHP', y=1.02, fontsize=24)
        ax_LDASKANTHPLNDASKANTHP.suptitle(f'{Dataset} - House {m}, LDA_SKANTHP vs LNDA_SKANTHP', y=1.02, fontsize=24)
        ax_LNDASTHPLNDASKANTHP.suptitle(f'{Dataset} - House {m}, LNDA_STHP vs LNDA_SKANTHP', y=1.02, fontsize=24)
        ax_LDASTHPLDASKANTHP.suptitle(f'{Dataset} - House {m}, LDA_STHP vs LDA_SKANTHP', y=1.02, fontsize=24)
        ax_LDALDASTHP.suptitle(f'{Dataset} - House {m}, LDA_HP vs LDA_STHP', y=1.02, fontsize=24)
        ax_LNDALNDASTHP.suptitle(f'{Dataset} - House {m}, LNDA_HP vs LNDA_STHP', y=1.02, fontsize=24)
        
        appliances=[]
        
        # Per-appliance evaluation
        for j in range(num_vertices):

            ax_subplot = ax.add_subplot(2, int(np.ceil(num_vertices / 2)), j + 1)
            ax_LDALNDA_subplot = ax_LDALNDA.add_subplot(2, int(np.ceil(num_vertices / 2)), j + 1)
            ax_LDASTHPLNDASTHP_subplot = ax_LDASTHPLNDASTHP.add_subplot(2, int(np.ceil(num_vertices / 2)), j + 1)
            ax_LDASKANTHPLNDASKANTHP_subplot = ax_LDASKANTHPLNDASKANTHP.add_subplot(2, int(np.ceil(num_vertices / 2)), j + 1)
            ax_LNDASTHPLNDASKANTHP_subplot = ax_LNDASTHPLNDASKANTHP.add_subplot(2, int(np.ceil(num_vertices / 2)), j + 1)
            ax_LDASTHPLDASKANTHP_subplot = ax_LDASTHPLDASKANTHP.add_subplot(2, int(np.ceil(num_vertices / 2)), j + 1)
            ax_LDALDASTHP_subplot = ax_LDALDASTHP.add_subplot(2, int(np.ceil(num_vertices / 2)), j + 1)
            ax_LNDALNDASTHP_subplot = ax_LNDALNDASTHP.add_subplot(2, int(np.ceil(num_vertices / 2)), j + 1)
            plots=[ax_subplot,ax_LDALNDA_subplot,ax_LDASTHPLNDASTHP_subplot,ax_LDASKANTHPLNDASKANTHP_subplot,ax_LNDASTHPLNDASKANTHP_subplot,ax_LDASTHPLDASKANTHP_subplot,ax_LDALDASTHP_subplot,ax_LNDALNDASTHP_subplot]

            # Set the title for each subplot
            for plot in plots:
                plot.set_title(Appliances[j], fontsize=18)
            y_truth = inputs[m][list(inputs[m].keys())[j + Len_of_inputs]].values
            if j==0:
                ax1_ap_subplot=ax_1ap.add_subplot(1, 1, j + 1)
                ax1_ap_subplot.fill_between(range(wind),y_truth[:wind], color='gray',alpha=0.3,label='Truth')
                ax1_ap_subplot.set_xlabel('Epochs (Hours)')
                ax1_ap_subplot.set_ylabel('Energy Level')
                
            keys=[]
            for key, value in All_Results.items():
                if 'THP' in key:
                    profiles=torch.sum(torch.squeeze(torch.stack(value.Omega[m],dim=0),dim=1)*value.rho[m],dim=-1)
                else:
                    
                    profiles =torch.sum(value.rho[m] * torch.normal(value.Omega[m, np.newaxis, :, :]), axis=-1)
                profiles_np = profiles.cpu().detach().numpy()
                
                y_true = inputs[m][list(inputs[m].keys())[j + Len_of_inputs]].values
                y_pred = Kmeans_mods[j].predict(profiles_np[:, j].reshape(-1,1))
                keys.append({key:y_pred})

                
                metrics_df = all_metrics(y_true, y_pred)
                metrics_df['Model'] = key
                metrics_df['Appliance'] = Appliances[j]
                metrics_df['House'] = m  # Add house index here
                metrics_df['Time']= value.time
                metrics_list.append(metrics_df)
                
                ax_subplot.plot(y_pred[:wind], label=f'{key} Prediction')

                # Plot prediction
                if j==0:
                    if key=="LDA_HP" or key=="LNDA_SKANTHP":
                        ax1_ap_subplot.plot(y_pred[:wind], label=f'{key} Prediction')
                        ax1_ap_subplot.legend()
                        ax1_ap_subplot.grid(True)

                if key=="LDA_HP"  or key=="LNDA_HP":    
                    ax_LDALNDA_subplot.plot(y_pred[:wind], label=f'{key} Prediction')

                if key=="LDA_STHP"  or key=="LNDA_STHP":    
                    ax_LDASTHPLNDASTHP_subplot.plot(y_pred[:wind], label=f'{key} Prediction')

                if key=="LDA_SKANTHP"  or key=="LNDA_SKANTHP":    
                    ax_LDASKANTHPLNDASKANTHP_subplot.plot(y_pred[:wind], label=f'{key} Prediction')

                if key=="LNDA_STHP"  or key=="LNDA_SKANTHP":    
                    ax_LNDASTHPLNDASKANTHP_subplot.plot(y_pred[:wind], label=f'{key} Prediction')

                if key=="LDA_STHP"  or key=="LDA_SKANTHP":    
                    ax_LDASTHPLDASKANTHP_subplot.plot(y_pred[:wind], label=f'{key} Prediction')

                if key=="LDA_HP"  or key=="LDA_STHP":    
                    ax_LDALDASTHP_subplot.plot(y_pred[:wind], label=f'{key} Prediction')

                if key=="LNDA_HP"  or key=="LNDA_STHP":    
                    ax_LNDALNDASTHP_subplot.plot(y_pred[:wind], label=f'{key} Prediction')
            
            # Plot 
            for plot in plots:
                plot.fill_between(range(wind),y_truth[:wind], color='gray',alpha=0.3,label='Truth')
                plot.legend()
                plot.grid(True)
                plot.set_xlabel('Epochs (Hours)')
                plot.set_ylabel('Energy Level')    

            appliances.append({Appliances[j]:keys})
        # Save all figures
        plot_path = os.path.join(parsed.data, 'figs')

        os.makedirs(plot_path, exist_ok=True)
        ax.savefig(f'{plot_path}/House{m}_ALL.png', bbox_inches='tight')
        ax_1ap.savefig(f'{plot_path}/House{m}_LDA_HP_vs_LNDA_SKANTHP.png', bbox_inches='tight')
        ax_LDALNDA.savefig(f'{plot_path}/House{m}_LDA_HP_vs_LNDA_HP.png', bbox_inches='tight')
        ax_LDASTHPLNDASTHP.savefig(f'{plot_path}/House{m}_LDA_STHP_vs_LNDA_STHP.png', bbox_inches='tight')
        ax_LDASKANTHPLNDASKANTHP.savefig(f'{plot_path}/House{m}_LDA_SKANTHP_vs_LNDA_SKANTHP.png', bbox_inches='tight')
        ax_LNDASTHPLNDASKANTHP.savefig(f'{plot_path}/House{m}_LNDA_STHP_vs_LNDA_SKANTHP.png', bbox_inches='tight')
        ax_LDASTHPLDASKANTHP.savefig(f'{plot_path}/House{m}_LDA_STHP_vs_LDA_SKANTHP.png', bbox_inches='tight')
        ax_LDALDASTHP.savefig(f'{plot_path}/House{m}_LDA_HP_vs_LDA_STHP.png', bbox_inches='tight')
        ax_LNDALNDASTHP.savefig(f'{plot_path}/House{m}_LNDA_HP_vs_LNDA_STHP.png', bbox_inches='tight')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        house.append(appliances)

        # Concatenate and collect metrics for each house
        all_metrics_df = pd.concat(metrics_list, ignore_index=True)
        all_metrics_df = all_metrics_df[['House', 'Model', 'Appliance', 'Time','MAE', 'MAPE', 'RMSE', 'MSE', 'FECA']]
        all_metrics_with_houses.append(all_metrics_df)

    # Final DataFrame across all houses
    final_metrics_df = pd.concat(all_metrics_with_houses, ignore_index=True)

    # Save DataFrame to CSV
    file_path = os.path.join(parsed.data, f"final_metrics_{parsed.iteration}.csv")
    file_path2 = os.path.join(parsed.data, "All_profiles.pickle")
    if not os.path.exists(file_path):
        final_metrics_df.to_csv(file_path, index=False)
    with open(file_path2, 'wb') as file:
        pickle.dump(all_houses, file)



if __name__ == '__main__':
    main()
# Standard Libraries
import datetime
import warnings
from collections import OrderedDict

# Data Science and Machine Learning Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# PyTorch Libraries
import torch

# Disable warnings
warnings.filterwarnings("ignore")

# Use CUDA if available
if torch.cuda.is_available():            
    torch.set_default_device('cuda')
    # print('Using CUDA:', torch.get_default_device())
torch.cuda.empty_cache()

# Other essential setup
# Place any other initial setup you need here


def extract_embeddings(inputs, Appliances, All_inputs, num_clusters=3, max_dims=3,):
    vocab = []
    trainset = []
    clfs = []
    X_trains = []
    y_trains = []
    testings = []
    D = len(inputs)

    for m in range(D):
        z = []
        inp = inputs[m]

        for i in range(len(inp[All_inputs[0]])):
            z.append("".join(str(inp[key][i]) for key in All_inputs))

        training = np.array([inp[key] for key in All_inputs], dtype='float').T
        testing = np.array([inp[appliance] for appliance in Appliances], dtype='float').T

        testings.append(testing)
        X_train, X_test, y_train, y_test = train_test_split(
            training, testing, test_size=0.33, random_state=42)
        X_trains.append(X_train)
        y_trains.append(y_train)

        clf = []
        for i in range(testing.shape[1]):
            model = KMeans(n_clusters=num_clusters).fit(y_train[:, i][:, np.newaxis])
            clf_model = make_pipeline(
                StandardScaler(),
                LinearSVC(random_state=0, tol=1e-5,penalty='l1')
            ).fit(X_train, model.predict(y_train[:, i][:, np.newaxis]).astype(int))
            clf.append(clf_model)
        clfs.append(clf)
        vocab += z
        trainset.append(z)

    # Deduplicate vocab
    vocab = list(OrderedDict.fromkeys(vocab))
    word_to_ix = {w: i for i, w in enumerate(vocab)}

    def seq_to_ix(seq, vocab=vocab):
        unk_idx = len(vocab)
        return np.array([word_to_ix.get(w, unk_idx) for w in seq])

    docs = list(map(seq_to_ix, trainset))

    # Compute input embedding representation
    training_tensor = torch.from_numpy(training)
    X = torch.sum(training_tensor, axis=1)

    
    Totals=[]
    for j in range(D):
        Total=[]
        for i in range(len(testing[0])):
            if(len((clfs[j][i].named_steps['linearsvc'].coef_))>=max_dims):
                Total.append(np.array(clfs[j][i].named_steps['linearsvc'].coef_[:max_dims]))
            else:
                Total.append(np.array([clfs[j][i].named_steps['linearsvc'].coef_[0] for z in range(max_dims)]))
        
        Total=np.array(Total)
        Totals.append(Total)
    Totals=np.array(Totals) 
    x=np.transpose(np.repeat((np.transpose(Totals,(1,0,2,3))@np.transpose((np.array(X_trains,dtype='float32')),(0,2,1)))[:,:,np.newaxis,:],len(training),axis=2),(1,0,3,2,4))
    W=torch.mean(torch.tensor(x),axis=[2,-2,-1]).T@torch.mean(torch.tensor(x),axis=[2,-2,-1])

    # Normalize
    tensor_min = W.min(dim=1, keepdim=True).values
    tensor_max = W.max(dim=1, keepdim=True).values
    W = (W - tensor_min) / (tensor_max - tensor_min)

    W_emb = W.cpu().numpy()
    A_emb = np.round(W_emb)
    x = torch.from_numpy(x.transpose(0, 3, 1, 2, 4)).to('cuda')

    return W_emb, A_emb, x, testings, y_trains, X,docs,vocab,

def Preprocess(Dataset,num_types,allhouses=None):
    if Dataset == 'REDD':
        Appliances=['kitchen','lights','elec','washer']
        inputs=['time','power1','power2']
        if allhouses==None:
            allhouses=2
    if Dataset == 'IRISE':
        Appliances=['TV','washing_drying','kitchen','fridge','heating']
        inputs=['time','power1']
        if allhouses==None:
            allhouses=2
    if Dataset == 'REFIT':
        Appliances=['Appliance1','Appliance2','Appliance3','Appliance4']
        inputs=['time','power1']
        if allhouses==None:
            allhouses=2
    if Dataset == 'UK_DALE':
        Appliances=['fridge','washer','dishwasher']
        inputs=['time','power1']
        if allhouses==None:
            allhouses=2
    Len_of_inputs=0  
    Dir=f"{Dataset}/"
    

    All_inputs=[]
    if 'time' in inputs:
        Len_of_inputs += 5
        All_inputs.extend(['year','month','week','weekday','hour'])
    if 'power1' in inputs:
        Len_of_inputs += 1
        All_inputs.extend(['power1'])
    if 'power2' in inputs:
        Len_of_inputs += 1
        All_inputs.extend(['power2'])
    if 'power3' in inputs:
        Len_of_inputs += 1
        All_inputs.extend(['power3'])
        
    num_vertices = len(Appliances)
    days={"Monday":2,"Tuesday":3,"Wednesday":4,"Thursday":5,"Friday":6,"Saturday":7,"Sunday":1}
    inputs=[]
    All_sequences=[]
    lowest_timestamp = 1000000000000
    for j in range(allhouses):
        data = pd.read_csv(Dir+"House"+str(j)+".csv")
        input={}
        input['time']=data.iloc[:,0]
        input['hour']=[]
        input['weekday']=[]
        input['month']=[]

        input['year']=[]
        input['week']=[]
        for i,t in enumerate(input['time']):
            # input['hour'].append(float(t[11:13]))
            input['hour'].append(int(datetime.datetime.strptime(t, '%Y-%m-%d %H:%M').hour)+1)

            input['weekday'].append(int(days[datetime.datetime.strptime(t[:10],'%Y-%m-%d').strftime('%A')]))
            input['month'].append(int(datetime.datetime.strptime(t[:10],'%Y-%m-%d').strftime('%m')))
            input['year'].append(datetime.datetime.strptime(t[:10], '%Y-%m-%d').year)

            input['week'].append(int(datetime.datetime.strptime(t[:10], '%Y-%m-%d').isocalendar()[1]))
        for key in ['power1', 'power2', 'power3']:
            if key in All_inputs:
                input[key] = data.iloc[:, 1]
 
        #outputs
        for idx, appliance in enumerate(Appliances, start=Len_of_inputs-4):
            input[appliance] = data.iloc[:, idx]
        
        # Convert to DataFrame
        df = pd.DataFrame(input)

        # Calculate average per hour
        avg_per_hour = df.groupby(All_inputs[:5])[All_inputs[5:] + Appliances].sum().reset_index()
        if len(avg_per_hour) < lowest_timestamp:
            lowest_timestamp = len(avg_per_hour)
        Kmeans_mods=[]
        avg_per_hour=avg_per_hour.head(500)
        inputs.append(avg_per_hour)
        for i in range(num_vertices):
            kmeans = KMeans(n_clusters=num_types, random_state=0).fit(np.array(avg_per_hour[Appliances[i]]).reshape(-1, 1))
            Kmeans_mods.append(kmeans)
            avg_per_hour[Appliances[i]] = kmeans.labels_
        avg_per_hour['time_since_start'] = avg_per_hour['hour'] + 24 * avg_per_hour['weekday'] + 7 * 24 * avg_per_hour['week']
        avg_per_hour['time_since_last_event'] = avg_per_hour['time_since_start'].diff().fillna(0)

        
        melted_df = avg_per_hour.melt(
        id_vars=['hour','time_since_start','time_since_last_event'],
        value_vars=Appliances,
        var_name='vertex',
        value_name='type_event'
        )

        appliance_map = {appliance: idx for idx, appliance in enumerate(Appliances)}
        melted_df['vertex'] = melted_df['vertex'].map(appliance_map)
        melted_df = melted_df.sort_values(by=['time_since_start', 'vertex']).reset_index(drop=True)

        Sequences=[]
        for i,week in enumerate(melted_df['time_since_start'].unique()):
            Sequences.append(melted_df[melted_df['time_since_start'] == week][['time_since_start', 'time_since_last_event','type_event','vertex']].reset_index(drop=True))
        for i in range(len(Sequences)):
            Sequences[i]=Sequences[i].to_dict(orient='records')
        All_sequences.append(Sequences)
    
    # W_emb, A_emb, x, testings, y_trains, X = extract_embeddings(inputs, Appliances, All_inputs)
    #trim all inputs to the same length lowest_timestamp
    for i in range(len(inputs)):
        inputs[i] = inputs[i][:lowest_timestamp]
        All_sequences[i] = All_sequences[i][:lowest_timestamp]


    # return All_sequences, num_vertices, allhouses, Appliances, inputs, Dataset, Len_of_inputs, All_inputs
        W_emb, A_emb, x, testings, y_trains, X,docs,vocab= extract_embeddings(inputs, Appliances, All_inputs)
    return Kmeans_mods,W_emb, A_emb, x, testings, y_trains, X, All_sequences, num_vertices, allhouses, Appliances, inputs, Dataset, Len_of_inputs,docs,vocab,num_types 


from Engine.NeuralEngine.KalmanNet import KalmanNet
from Engine.NeuralEngine.Pipeline import Pipeline
from Engine.ModelFrame import SystemModel
from Engine.Filters.EKF import EKFTest
from datetime import datetime
import config as config
import torch

m: int = 9
n: int = 1000

print("Pipeline Start")
################
# Get Time ###
################
print("Current Time =", datetime.today().strftime("%m.%d.%y") + "_" + datetime.now().strftime("%H:%M:%S"))

###################
#     Settings
###################

args = config.general_settings()
# dataset parameters
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
args.T = 20
args.T_test = 20
# settings for KalmanNet
args.in_mult_KNet = 40
args.out_mult_KNet = 5

# training parameters
args.use_cuda = True  # use GPU or not
args.n_steps = 2000
args.n_batch = 100
args.lr = 1e-4
args.wd = 1e-4
args.CompositionLoss = True
args.alpha = 0.5

if args.use_cuda:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

offset = 0
sequential_training = False
path_results = './Output/KNet/'
DatafolderName = './DTO/pm25_weer.csv'
r2 = torch.tensor([1e-3])  # [10, 1, 0.1, 0.01, 1e-3]
vdB = 0  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)

Q_structure: torch.tensor = torch.eye(m)
R_structure: torch.tensor = torch.eye(m)
f: torch.tensor = torch.eye(m)
h: torch.tensor = torch.eye(m)
m1x_0: torch.tensor = torch.eye(m)
m2x_0: torch.tensor = torch.eye(m)

Q = q2[0] * Q_structure
R = r2[0] * R_structure

print("1/r2 [dB]: ", 10 * torch.log10(1 / r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1 / q2[0]))

traj_resultName = ['traj_lorDT_NLobs_rq3030_T20.pt']
dataFileName = ['data_lor_v0_rq3030_T20.pt']

#########################################
#              Load data
#########################################

sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)

print("Data Load")
print(dataFileName[0])
[
    train_input_long,
    train_target_long,
    cv_input,
    cv_target,
    test_input,
    test_target,
    _,
    _,
    _
] = torch.load(DatafolderName + dataFileName[0], map_location=device)
train_target = train_target_long[:, :, 0:args.T]
train_input = train_input_long[:, :, 0:args.T]
cv_target = cv_target[:, :, 0:args.T]
cv_input = cv_input[:, :, 0:args.T]

print("Train set size:", train_target.size())
print("Cross-Validation set size:", cv_target.size())
print("Test set size:", test_target.size())

########################
#   Evaluate Filters
########################

#   Evaluate EKF full
print("Evaluate EKF")
[
    MSE_EKF_linear_arr,
    MSE_EKF_linear_avg,
    MSE_EKF_dB_avg,
    EKF_KG_array,
    EKF_out
] = EKFTest(args, sys_model, test_input, test_target)

##########################
#   Evaluate KalmanNet
##########################

#   Build Neural Network
print("KalmanNet start")
KalmanNet_model = KalmanNet()
KalmanNet_model.NNBuild(sys_model, args)

# Train Neural Network
KalmanNet_Pipeline = Pipeline(folderName="KNet", modelName="KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)

print("Number of parameters for KNet:", sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
KalmanNet_Pipeline.setTrainingParams(args)
print("Composition Loss:", args.CompositionLoss)

[
    MSE_cv_linear_epoch,
    MSE_cv_dB_epoch,
    MSE_train_linear_epoch,
    MSE_train_dB_epoch
] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)

# Test Neural Network
[
    MSE_test_linear_arr,
    MSE_test_linear_avg,
    MSE_test_dB_avg,
    knet_out,
    RunTime
] = KalmanNet_Pipeline.NNTest(sys_model,
                              test_input,
                              test_target,
                              path_results)

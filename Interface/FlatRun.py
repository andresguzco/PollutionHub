from Engine.KalmanNet import KalmanNet
from Engine.Pipeline import Pipeline
from Engine.ModelFrame import SystemModel
from Engine.EKF import EKFTest
from datetime import datetime
import config as config
import torch


class RunFlow(object):
    def __init__(self):
        self.Args = self.InitSettings()
        self.Device = self.SetupDevice()
        self.Database = dict()

    @staticmethod
    def GetTime():
        return datetime.today().strftime("%m.%d.%y") + "_" + datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def InitSettings():
        args = config.general_settings()
        args.T = 20
        args.lr = 1e-4
        args.wd = 1e-4
        args.N_T = 200
        args.N_E = 1000
        args.N_CV = 100
        args.alpha = 0.5
        args.T_test = 20
        args.n_batch = 100
        args.n_steps = 2000
        args.use_cuda = True
        args.in_mult_KNet = 40
        args.out_mult_KNet = 5
        args.CompositionLoss = True
        return args

    def SetupDevice(self):
        if self.Args.use_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        elif self.Args.use_cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please set args.use_cuda = False")
        else:
            return torch.device('cpu')

    def LoadData(self, m: int, n: int) -> None:
        DatafolderName = './DTO/pm25_weer.csv'
        dataFileName = ['data_lor_v0_rq3030_T20.pt']
        r2 = torch.tensor([1e-3])
        vdB = 0
        v = 10 ** (vdB / 10)
        q2 = torch.mul(v, r2)

        Q_structure = torch.eye(m)
        R_structure = torch.eye(m)
        f = torch.eye(m)
        h = torch.eye(m)

        Q = q2[0] * Q_structure
        R = r2[0] * R_structure

        self.Database["System Model"] = SystemModel(f, Q, h, R, self.Args.T, self.Args.T_test, m, n)

        [
            train_input_long,
            train_target_long,
            cv_input,
            cv_target,
            self.Database["Test Input"],
            self.Database["Test Target"],
            _,
            _,
            _
         ] = torch.load(DatafolderName + dataFileName[0], map_location=self.Device)

        self.Database["Train Target"] = train_target_long[:, :, 0:self.Args.T]
        self.Database["Train Input"] = train_input_long[:, :, 0:self.Args.T]

        self.Database["CV Target"] = cv_target[:, :, 0:self.Args.T]
        self.Database["CV Input"] = cv_input[:, :, 0:self.Args.T]

        return None

    def EvaluateFilters(self, test_input, test_target, sys_model):
        return EKFTest(self.Args, sys_model, test_input, test_target)

    def EvaluateKalmanNet(self, cv_input, cv_target, train_input, train_target, test_input, test_target, sys_model):
        path_results = './Output/KNet/'
        KalmanNet_model = KalmanNet()
        KalmanNet_model.NNBuild(sys_model, self.Args)

        KalmanNet_Pipeline = Pipeline(folderName="KNet", modelName="KalmanNet")
        KalmanNet_Pipeline.setssModel(sys_model)
        KalmanNet_Pipeline.setModel(KalmanNet_model)
        KalmanNet_Pipeline.setTrainingParams(self.Args)

        KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
        return KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)

    def __getitem__(self, item: str):
        if item is not int:
            print("Please provide a valid 'str' item to retrieve from the model's database.")
        return self.Database[item]


def main():
    m = 9
    n = 1000
    Printer: bool = True

    flow = RunFlow()

    print("Pipeline Start")
    print("Current Time =", flow.GetTime())

    flow.LoadData(m, n)
    print("Data Load")
    print("Train set size:", flow["Train Target"].size())
    print("Cross-Validation set size:", flow["CV Target"].size())
    print("Test set size:", flow["Test Target"].size())

    print("Evaluate EKF")
    EKF_results = flow.EvaluateFilters(flow["Test Input"], flow["Test Target"], flow["System Model"])

    print("KalmanNet start")
    KNet_results = flow.EvaluateKalmanNet(
        flow["CV Input"], flow["CV Target"],
        flow["Train Input"], flow["Train Target"],
        flow["Test Input"], flow["Test Target"],
        flow["System Model"]
    )
    print("KalmanNet Testing Complete")

    if Printer:
        print(EKF_results)
        print(KNet_results)


if __name__ == "__main__":
    main()

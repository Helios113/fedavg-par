import copy
import torch
from client_a import Client
from models import SLC_co, MLP
from dataset import FedDataset, get_data, DataInfo
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from datetime import datetime
import yaml
import shutil

torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser(description="DM-FedAvg")
# Optional argument

parser.add_argument(
    "--paramPath",
    required=True,
    action="store",
    type=str,
    help="yaml file with modality parameters",
)

args = parser.parse_args()

dirs = os.path.split(args.paramPath)
save_path = os.path.dirname(args.paramPath)
parent_path = os.path.dirname(save_path)
og_save_path = os.path.join("/data",os.path.dirname(args.paramPath))
# model parameters
params_loc_path = os.path.join(save_path, "paramsLoc")
params_glob_path = os.path.join(save_path, "paramsGlob")
loss_train_path = os.path.join(save_path, "loss_train.txt")
loss_test_path = os.path.join(save_path, "loss_test.txt")
f1_test_path = os.path.join(save_path, "f1_test.txt")
acc_test_path = os.path.join(save_path, "acc_test.txt")
info_path = os.path.join(save_path, "info.txt")

glob_file = os.path.join(parent_path, "glob")
loc_file = os.path.join(parent_path, "loc")
lam_file = os.path.join(parent_path, "lam.txt")


"""
Fiels
"""
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(os.path.join(save_path,"corr_fuse")):
    os.makedirs(os.path.join(save_path,"corr_fuse"))
if not os.path.exists(os.path.join(save_path,"corr_gran")):
    os.makedirs(os.path.join(save_path,"corr_gran"))

if not os.path.exists(og_save_path):
    os.makedirs(og_save_path)
if not os.path.exists(os.path.join(og_save_path,"corr_fuse")):
    os.makedirs(os.path.join(og_save_path,"corr_fuse"))
if not os.path.exists(os.path.join(og_save_path,"corr_gran")):
    os.makedirs(os.path.join(og_save_path,"corr_gran"))

             

lss_train_f = open(loss_train_path, "w+")
lss_test_f = open(loss_test_path, "w+")

f1_test_f = open(f1_test_path, "w+")
acc_test_f = open(acc_test_path, "w+")

info_f = open(info_path, "w+")
test_freq = 1


# Determine hardware availability
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

# Test parameters


all_mod = {
    "all": [0, 1, 2, 5, 6, 7, 14, 15, 16, 8, 9, 10, 17, 18, 19, 11, 12, 13, 20, 21, 22],
    "all_s":[5, 6, 7, 8, 9, 10,11, 12, 13,14,15,16, 17, 18, 19,20, 21, 22],
    "d1": [5, 6, 7, 8, 9, 10,11, 12, 13],
    "d2": [14, 15, 16, 17, 18, 19,20, 21, 22],
    "acc": [0, 1, 2, 5, 6, 7, 14, 15, 16],
    "acc3": [0, 1, 2],
    "acc1": [5, 6, 7],
    "acc2": [14, 15, 16],
    "gyro": [8, 9, 10, 17, 18, 19],
    "gyro1": [8, 9, 10],
    "gyro2": [17, 18, 19],
    "mag": [ 11, 12, 13, 20, 21, 22],
    "mag1": [11, 12, 13,],
    "mag2": [ 20, 21, 22],
}
try:
    with open(args.paramPath, "r") as file:
        yml_file = yaml.safe_load(file)
        modalities = yml_file["modalities"]

        hidden_dims = yml_file["hidden_dims"]
        num_clients = yml_file["n_clients"]
        num_sets_train = yml_file["n_sets_train"]
        num_sets_test = yml_file["n_sets_test"]

        batch_size = yml_file["batch_size"]
        federatedGlob = yml_file["fedGlob"]
        federatedLoc = yml_file["fedLoc"]

        learning_rate = float(yml_file["lr"])
        optimizer = yml_file["optim"]

        alpha = 1
        alpha_per_modality = False
        lg_frac = yml_file["lg_frac"]
        rounds = yml_file["rounds"]
        local_epochs = yml_file["epochs"]
        iid = yml_file["iid"]
        penalty = yml_file["penalty"]
        classes_per_client_training = yml_file["classes_per_client_training"]
        classes_per_client_testing = yml_file["classes_per_client_testing"]
        train_data_portion = yml_file["train_data_portion"]
        test_data_portion = yml_file["test_data_portion"]
        mlp_dims = yml_file["mlp_dims"]
        transient_dim = mlp_dims[0]
        output_dim=mlp_dims[-1]
        train = yml_file["train"]
        data_path = yml_file["data_path"]
        evaluate = yml_file["eval"]
        corr = yml_file["corr"]
        sig = yml_file["sig"]
        diff = yml_file["diff"]
        
        
except Exception as e:
    print(e)
except:
    print("no yaml file given")
    exit()

print(r"{:-^30}".format("PID"), file=info_f)
print(r"{txt:<20}:{val}".format(txt="pid", val=os.getpid()), file=info_f)
print(yml_file, file=info_f)
print(modalities, file=info_f)
print(datetime.now(), file=info_f)



data_train, data_test = get_data(
    data_path,
    num_clients,
    iid,
    DataInfo(
        train_data_portion,
        test_data_portion,
        classes_per_client_training,
        classes_per_client_testing,
        num_sets_train,
        num_sets_test,
    ),
)
clients = []

uni_loc = SLC_co(all_mod, hidden_dims, transient_dim, penalty, sig, diff)
uni_glob = MLP(mlp_dims)
if os.path.exists(glob_file):
    uni_glob.load_state_dict(torch.load(glob_file))
else:
    torch.save(uni_glob.state_dict(),glob_file)
if os.path.exists(loc_file):
    if evaluate:
        b = torch.load(loc_file)
        uni_loc.get_submodule("layer_list").load_state_dict(b, strict=False)
        b = np.loadtxt(lam_file).reshape(1)
        b = torch.Tensor(b)
        uni_loc.get_submodule("lam").load_state_dict({"0" : b})
    else:
        b = torch.load(loc_file)
        uni_loc.get_submodule("layer_list").load_state_dict(b, strict=False)
else:
    torch.save(uni_loc.state_dict(),loc_file)

for i in range(num_clients):
    glob_mod = MLP(mlp_dims)
    local_mod = SLC_co(modalities[i], hidden_dims, transient_dim, penalty, sig, diff)
    
    s_dict = {}
    local_dict = uni_loc.state_dict()
    for k in local_mod.state_dict():
        s_dict[k] = copy.deepcopy(local_dict[k])
    local_mod.load_state_dict(s_dict)
    s_dict = {}
    global_dict = uni_glob.state_dict()
    for k in glob_mod.state_dict():
        s_dict[k] = copy.deepcopy(global_dict[k])
    glob_mod.load_state_dict(s_dict)

    clients.append(
        Client(
            glob_mod,
            local_mod,
            DataLoader(
                FedDataset(data_train[i % num_sets_train], device),
                batch_size=batch_size,
                shuffle=True,
            ),
            DataLoader(
                FedDataset(data_test[i % num_sets_test], device),
                batch_size=32,
                shuffle=True,
            ),
            local_epochs,
            learning_rate,
            learning_rate,
            optimizer,
            device=device,
        )
    )

print("number of test batches:",np.ceil(len(data_test[0])/32), file=info_f)
info_f.flush()

last_entry = 0
performance = np.zeros((num_clients, 2, rounds))
acc = np.zeros((num_clients, 2, rounds))
loss = np.zeros((num_clients, 2, rounds))
init_time = datetime.now()
max_f1 = np.zeros(num_clients)



glob_list = [None]*num_clients
loc_list = [None]*num_clients
for round in range(rounds):
    last_time = datetime.now()


    # Count of encounters of each param
    w_loc_tmp_count = None
    if round > (1 - lg_frac) * rounds:
        federatedGlob = True
        train = train - 1

    for client in range(num_clients):
        if not evaluate:
            w_glob_ret, w_local_ret, loss[client,
                                            0, round] = clients[client].train(train, round)
        (
            performance[client, 0, round],
            performance[client, 1, round],
            acc[client, 0, round],
            acc[client, 1, round],vec,grans
        ) = clients[client].test()
        for w in range(12):
            c_vec = torch.vstack(vec[w])
            np.savetxt(os.path.join(save_path,"corr_fuse", "featue_vec_r_{}_d_{}_{}.txt".format(round, client,w)), c_vec.cpu().numpy().mean(axis=0))
            for q in range(len(grans[0])):
                c_vec = torch.vstack(grans[w][q])
                np.savetxt(os.path.join(save_path,"corr_gran", "featue_vec_r_{}_d_{}_{}_{}.txt".format(round, client,q,w)), c_vec.cpu().numpy().mean(axis=0))

        if not evaluate:
            if federatedGlob:
                glob_list[client] = copy.deepcopy(w_glob_ret)
            if federatedLoc:
                loc_list[client] = copy.deepcopy(w_local_ret)

    for client in range(num_clients):
        if performance[client, 0, round] > max_f1[client]:
            max_f1[client] = performance[client, 0, round]
            torch.save(
                clients[client].get_params()[0], params_glob_path +
                f"{client}.mp"
            )
            torch.save(
                clients[client].get_params()[1], params_loc_path +
                f"{client}.mp"
            )
            a = clients[client].get_params()[2]['0']
            np.savetxt(params_loc_path + f"labda_{client}.txt", a.cpu().numpy())

           
    train_loss = np.char.mod("%f", loss[:, 0, round].reshape(-1))
    train_loss = ",".join(train_loss)

    mean_std = np.char.mod("%f", performance[:, :, round].reshape(-1))
    mean_std = ",".join(mean_std)
    
    mean_std1 = np.char.mod("%f", acc[:, :, round].reshape(-1))
    mean_std1 = ",".join(mean_std1)

    test_loss = np.char.mod("%f", loss[:, 1, round].reshape(-1))
    test_loss = ",".join(test_loss)
    print(
        r"{},{},{},{}".format(
            datetime.now() - init_time, datetime.now() - last_time, round, train_loss
        ),
        file=lss_train_f,
    )
    print(
        r"{},{},{},{} ".format(
            datetime.now() - init_time, datetime.now() - last_time, round, test_loss
        ),
        file=lss_test_f,
    )
    print(
        r"{},{},{},{} ".format(
            datetime.now() - init_time, datetime.now() - last_time, round, mean_std
        ),
        file=f1_test_f,
    )
    print(
        r"{},{},{},{} ".format(
            datetime.now() - init_time, datetime.now() - last_time, round, mean_std1
        ),
        file=acc_test_f,
    )
    if evaluate:
        exit()
    # get weighted average for global weights
    for devs in corr:
        w_glob_tmp = None
        w_loc_tmp = None
        for l in devs:
            if federatedGlob:
                if w_glob_tmp is None:
                    w_glob_tmp = copy.deepcopy(glob_list[l])
                    for k in w_glob_tmp:
                        w_glob_tmp[k] = w_glob_tmp[k].unsqueeze(0)
                else:
                    for k in w_glob_tmp:
                        w_glob_tmp[k] = torch.cat((w_glob_tmp[k], glob_list[l][k].unsqueeze(0)), dim=0)
            if federatedLoc:
                if w_loc_tmp is None:
                    w_loc_tmp = {}
                    w_loc_tmp_count = {}
                for k in loc_list[l].keys():
                    if k not in w_loc_tmp:
                        w_loc_tmp[k] =  loc_list[l][k]
                        w_loc_tmp_count[k] = 1
                    else:
                        w_loc_tmp[k] +=  loc_list[l][k]
                        w_loc_tmp_count[k] += 1
        if federatedGlob:
            for k in w_glob_tmp.keys():
                w_glob_tmp[k] = torch.mean(w_glob_tmp[k], 0, False).squeeze(0)
        if federatedLoc:
            for k in w_loc_tmp.keys():
                w_loc_tmp[k] = torch.div(w_loc_tmp[k], w_loc_tmp_count[k])
        for l in devs:
            clients[l].load_params(w_glob_tmp, w_loc_tmp)
        
        
    # copy weights to each client based on mode

    if round%10==0:
        lss_train_f.flush()
        lss_test_f.flush()
        f1_test_f.flush()
        acc_test_f.flush()

dest = shutil.move(os.path.join(save_path,"corr_fuse"), os.path.join(og_save_path,"corr_fuse"))
dest = shutil.move(os.path.join(save_path,"corr_gran"), os.path.join(og_save_path,"corr_gran"))

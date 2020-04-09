import os
import torch
import json

class Logger:
    def __init__(self, methods, env, args):
        self.methods = methods
        self.env = env
        self.args = args
        self.t = 0

        self.init()
    
    def init(self):
        for method in self.methods:
            if method == "stdout":
                pass
            elif method == "tensorboard":
                import tensorboardX
                self.writer = tensorboardX.SummaryWriter(comment=self.args.name)
            elif method == "wandb":
                import wandb
                wandb.init(project=self.args.wandb_name, config=vars(self.args))
            else:
                raise Exception("Unknown loggin method: %s" % method)
        
    def log_scalars(self, stats):
        self.t += 1
        for method in self.methods:
            if method == "stdout" and self.t % self.args.log_steps == 0:
                print("=== Log ===")
                for key, value in stats.items():
                    print("- %s: %s" % (key, value))
                print("===========")
            if method == "tensorboard":
                for key, value in stats.items():
                    self.writer.add_scalar(key, value, self.t)
            if method == "wandb":
                import wandb
                wandb.log(stats, step=self.t)

    
    def log_results(self, results):
        for method in self.methods:
            if method == "stdout":
                print("==== RESULTS ====")
                for key, value in results.items():
                    print("- %s: %f" % (key, value))
                print("=================")
            if method == "tensorboard":
                self.writer.add_hparams(vars(self.args), results)
            if method == "wandb":
                import wandb
                wandb.log(results)
    
    def save_model(self, policy):
        save_path = "./checkpoints/"+self.args.name+"/"
        os.mkdir(save_path)

        torch.save(policy.state_dict(), save_path+"model.pth")
        open(save_path+"args.json", "w").write(json.dumps(vars(self.args)))
        for method in self.methods:
            if method == "wandb":
                import wandb
                wandb.save(save_path+"model.pth")
                wandb.save(save_path+"args.json")
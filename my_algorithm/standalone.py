from flgo.algorithm.fedbase import BasicServer, BasicClient
import copy
import torch
from flgo.utils import fmodule
import collections


class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.model = [copy.deepcopy(self.model) for i in range(10)]
        self.sample_option = 'full'
        self.init_algo_para({'train_on_all': True})

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        self.gv.logger.info("--------------Initial Evaluation--------------")
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        self.model = models
        self.gv.logger.log_once()
        self.gv.logger.info("=================End==================")
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return


    def iterate(self):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
        :return
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        self.model = self.communicate(self.selected_clients)['model']

        return len(self.model)>0

    def pack(self, client_id, mtype=0, *args, **kwargs):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return {
            "model" : copy.deepcopy(self.model[client_id]),
        }

    def global_test(self, dataflag='valid'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            dataflag: choose train data or valid data to evaluate
        :return
            metrics: a dict contains the lists of each metric_value of the clients
        """
        all_metrics = collections.defaultdict(list)
        for c, model in zip(self.clients, self.model):
            client_metrics = c.test(model, dataflag)

            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)

        return all_metrics

    def test(self, model=None, flag='test'):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        if model is None: model=self.model

        all_metrics = collections.defaultdict(list)
        for i in range(10):
            model_ = model[i]
            if self.test_data:
                client_metrics = self.calculator.test(model_, self.test_data, batch_size = self.option['test_batch_size'], num_workers = self.option['num_workers'], pin_memory = self.option['pin_memory'])

                for met_name, met_val in client_metrics.items():
                    all_metrics[met_name].append(met_val)
        return all_metrics

class Client(BasicClient):

    def initialize(self):
        if self.train_on_all:
            self.train_data.indices = self.train_data.indices + self.valid_data.indices
        return



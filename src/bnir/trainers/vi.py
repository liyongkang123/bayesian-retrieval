from bnir.trainers.base import TevatronTrainer


class BayesianTevatronTrainer(TevatronTrainer):
    def __init__(self, base_qry_model, base_psg_model, prior_model, **kwargs):
        super().__init__(**kwargs)
        self.base_qry_model = base_qry_model
        self.base_qry_model.to(self.model.qry_model.mu.device)
        self.base_psg_model = base_psg_model
        self.base_psg_model.to(self.model.psg_model.mu.device)
        self.prior_model = (
            prior_model  # Prior model is the same for both queries and passages.
        )
        self.prior_model.to(self.model.qry_model.mu.device)

    def compute_loss(self, model, inputs):
        query, passage = inputs
        return model(
            query,
            passage,
            self.base_qry_model,
            self.base_psg_model,
            self.prior_model,
        ).loss

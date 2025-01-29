class SentenceTransformerFinetuner():
    def __init__(self, model, tm, graph, epochs, warmup_steps):
        self.model = model
        self.tm = tm
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.graph = graph

    def finetune(self, triplets):
        train_dataset = self.tm.enrich_triplets(triplets)
        self.graph.sentencetransformer.finetune(train_dataset, self.epochs, self.warmup_steps)
        self.graph.update_encodings()
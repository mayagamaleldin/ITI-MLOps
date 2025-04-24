from mlflow.pyfunc import PythonModel, PythonModelContext

class ModelWrapper(PythonModel):
    def load_context(self, context: PythonModelContext):
        import pickle
        self._encoder = pickle.loads(context.artifacts["encoder"])
        self._model = pickle.loads(context.artifacts["model"])

    def predict(self, context: PythonModelContext, data):
        return self._model.predict(data)
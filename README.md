# refineGenerativeShapeGMM

SOMETHING

---

## 📜 Overview

Something 

✅ **Features**:

* Feature 1
* Feature 2

---

## 📦 Installation

Directions to install

```bash
git clone https://github.com/mccullaghlab/vonMisesMixtureModel.git
cd vonMisesMixtureModel
pip install .
```

**Dependencies**:

* `numpy`
* `scipy`
* `shapeGMM`
* `pytest` (for running tests)

---

## 🧐 Usage

```python

```

---

## 🧠 API Overview

### `???`

Initialize the mixture model.

| Parameter      | Description                                   |
| -------------- | --------------------------------------------- |
| `n_components` | Number of clusters                            |
| `small_lambda` | Use small lambda approximation (default True) |
| `max_iter`     | Maximum EM iterations                         |
| `tol`          | Convergence threshold for log-likelihood      |
| `device`       | 'cuda' or 'cpu'                               |
| `verbose`      | Print progress during fitting                 |

---

### 🔧 Key Methods

| Method                        | Description                                            |
| ----------------------------- | ------------------------------------------------------ |
| `fit(data)`                   | Fit model to angular data of shape `(N, 2)`            |
| `predict(data)`               | Predict cluster assignments and compute log-likelihood |
| `ln_pdf(data)`                | Log-density under the fitted model                     |
| `pdf(data)`                   | Probability density under the fitted model             |
| `aic(data)`                   | Akaike Information Criterion                           |
| `bic(data)`                   | Bayesian Information Criterion                         |
| `icl(data)`                   | Integrated Complete Likelihood                         |
| `plot_scatter_clusters(data)` | Visualize 2D clusters                                  |

---

## 🧬 Applications

* App 1
* App 2

---

## 🛠️ Testing

To run the unit tests:

```bash
pytest tests/
```

---

## 📚 References


---

## 🙌 Contributing

Contributions are welcome! Please open an issue or pull request if you'd like to contribute. A `CONTRIBUTING.md` will be added soon.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.


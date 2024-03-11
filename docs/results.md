# Results

Please find our publication at the following link: https://doi.org/mkfv.

If you use EDS-Pseudo, please cite us as below:

```
@article{eds_pseudo,
  title={Development and validation of a natural language processing algorithm to pseudonymize documents in the context of a clinical data warehouse},
  author={Tannier, Xavier and Wajsb{\"u}rt, Perceval and Calliger, Alice and Dura, Basile and Mouchet, Alexandre and Hilka, Martin and Bey, Romain},
  journal={Methods of Information in Medicine},
  year={2024},
  publisher={Georg Thieme Verlag KG}
}
```

To inspect the results for the latest version of our system, please refer to the
[latest release](/eds-pseudo/latest/results) page.

<!--

You will find below some of the results presented in the article, as well as interactive charts.

## General results

--8<-- "docs/assets/figures/ml_vs_rb_table.html"

*If you have trouble seeing the chart, please refresh the page.*

```vegalite
{
  "schema-url": "../assets/figures/label_scores.json"
}
```

---

## Impact of the language model

--8<-- "docs/assets/figures/bert_ablation_table.html"


*If you have trouble seeing the chart, please refresh the page.*

```vegalite
{
  "schema-url": "../assets/figures/bert_ablation.json"
}
```

---

## Impact of the PDF extraction step

--8<-- "docs/assets/figures/pdf_comparison_table.html"


---

## Impact of the number of training examples

*If you have trouble seeing the chart, please refresh the page.*

```vegalite
{
  "schema-url": "../assets/figures/limit_ablation.json"
}
```

## Impact of the missing document types

*If you have trouble seeing the chart, please refresh the page.*

```vegalite
{
  "schema-url": "../assets/figures/doc_type_ablation.json"
}
```

-->

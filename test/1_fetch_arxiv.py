import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fetch_arxiv import fetch_papers

if __name__ == "__main__":
    results = fetch_papers(["graph neural networks healthcare"], max_results=10)
    print(f"Fetched {len(results)} papers.")
    # print(results[0])

'''
{
  "title": "Graph-in-Graph (GiG): Learning interpretable latent graphs in\n  non-Euclidean domain for biological and healthcare applications",
  "authors": [
    "Kamilia Mullakaeva",
    "Luca Cosmo",
    "Anees Kazi",
    "Seyed-Ahmad Ahmadi",
    "Nassir Navab",
    "Michael M. Bronstein"
  ],
  "summary": "Graphs are a powerful tool for representing and analyzing unstructured,\nnon-Euclidean data ubiquitous in the healthcare domain. Two prominent examples\nare molecule property prediction and brain connectome analysis. Importantly,\nrecent works have shown that considering relationships between input data\nsamples have a positive regularizing effect for the downstream task in\nhealthcare applications. These relationships are naturally modeled by a\n(possibly unknown) graph structure between input samples. In this work, we\npropose Graph-in-Graph (GiG), a neural network architecture for protein\nclassification and brain imaging applications that exploits the graph\nrepresentation of the input data samples and their latent relation. We assume\nan initially unknown latent-graph structure between graph-valued input data and\npropose to learn end-to-end a parametric model for message passing within and\nacross input graph samples, along with the latent structure connecting the\ninput graphs. Further, we introduce a degree distribution loss that helps\nregularize the predicted latent relationships structure. This regularization\ncan significantly improve the downstream task. Moreover, the obtained latent\ngraph can represent patient population models or networks of molecule clusters,\nproviding a level of interpretability and knowledge discovery in the input\ndomain of particular value in healthcare.",
  "published": "2022-04-01T10:01:37Z",
  "updated": "2022-04-01T10:01:37Z",
  "link": "http://arxiv.org/abs/2204.00323v1"
}
'''
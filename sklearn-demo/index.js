async function main() {
  let pyodide = await loadPyodide();
  await pyodide.loadPackage('numpy');
  await pyodide.loadPackage('matplotlib');
  await pyodide.loadPackage('scikit-learn');
  await pyodide.runPython(`
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from js import document
    from sklearn.decomposition import PCA
    
    my_str = "teste"
    
    rng = np.random.RandomState(0)
    n_samples = 500
    cov = [[3, 3], [3, 4]]
    X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
    pca = PCA(n_components=2).fit(X)
    
    f = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="samples")
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp = comp * var  # scale component by its variance explanation power
        plt.plot(
            [0, comp[0]],
            [0, comp[1]],
            label=f"Component {i}",
            linewidth=5,
            color=f"C{i + 2}",
        )
    plt.legend()
    plt.show()


    # ordinary function to create a div
    def create_root_element1(self):
        div = document.createElement('div')
        document.body.appendChild(div)
        return div

    #ordinary function to find an existing div
    #you'll need to put a div with appropriate id somewhere in the document
    def create_root_element2(self):
        return document.getElementById('figure1')

    #override create_root_element method of canvas by one of the functions above
    f.canvas.create_root_element = create_root_element1.__get__(
        create_root_element1, f.canvas.__class__)

    f.canvas.show()
    `)

    document.getElementById("textfield").innerText = pyodide.globals.my_str;
}

main();
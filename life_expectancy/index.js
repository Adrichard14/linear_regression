async function main() {
  let pyodide = await loadPyodide();
  await pyodide.loadPackage("micropip")
  await pyodide.loadPackage('numpy');
  await pyodide.loadPackage('matplotlib');
  await pyodide.loadPackage('scikit-learn');
  await pyodide.loadPackage('pandas');
  await pyodide.runPython(`
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import io

    import asyncio
    from js import console, fetch

    from pyodide.http import open_url
    url = 'https://somosship.com.br/explica/adrian/population.csv'
    url_content = open_url(url)
    data = pd.read_csv(url_content)

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    def calc(slope, intercept, hours):
        return slope*hours+intercept


    # df = pd.DataFrame(data, columns=['data', 'expectativa', 'porcetagem', ''])
    unformatted_y = data['data']
    y = []
    for i in unformatted_y:
        y.append(int(i.split('-')[0]))
    x = data['expectativa'].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=23)

    # x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)


    # x_test = np.array(x_test).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)

    a = linear_regression.intercept_
    b = linear_regression.coef_

    # y_pred_train = a*x_train*b
    y_pred_train = linear_regression.predict(x_train)
    y_pred_test = linear_regression.predict(x_test)
    # print(y_pred_train)

    # plt.scatter(x, y)
    plt.scatter(x_test, y_test)
    plt.scatter(x_test, y_pred_test, color='red')

    # score = calc(linear_regression.coef_, linear_regression.intercept_, 2050)
    print(score)
    # plt.ylim(-2, 2)
    # plt.ylabel('Data', )
    # plt.xlabel('Expectativa de vida')
    plt.yticks((1962, 1982, 1992, 2002, 2012, 2022))

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
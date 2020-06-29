Use Case 3: Customer Interest Clustering
========================================

This is a tutorial on how to perform customer clustering based on the
interests and purcahses of customers.

Marketing teams frequently want to know who how to appeal to their customers.

We’ll show how graph analytics can be used to gain insights about
the interests of customers by finding communities of customers who’ve
bought similar products.

We’lll accomplish this by creating a bipartite graph of customers and
products, using a graph projection to create a graph of customers linked
to other customers who’ve bought the same product, and using Louvain
community detection to find the communities.

We’ll be using ecommerce transaction data from a U.K. retailer provided
by the University of California, Irvine. The data can be found
`here <https://www.kaggle.com/carrie1/ecommerce-data>`__.

Data Preprocessing
==================

Let’s first look at the data.

First, we’ll need to import some libraries.

.. code:: python

    >>> import metagraph as mg
    >>> import pandas as pd
    >>> import networkx as nx

Let’s see what the data looks like.

.. code:: python

    >>> RAW_DATA_CSV = './data.csv' # https://www.kaggle.com/carrie1/ecommerce-data
    >>> data_df = pd.read_csv(RAW_DATA_CSV, encoding="ISO-8859-1")
    >>> data_df.head()

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>InvoiceNo</th>
          <th>StockCode</th>
          <th>Description</th>
          <th>Quantity</th>
          <th>InvoiceDate</th>
          <th>UnitPrice</th>
          <th>CustomerID</th>
          <th>Country</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>536365</td>
          <td>85123A</td>
          <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
          <td>6</td>
          <td>12/1/2010 8:26</td>
          <td>2.55</td>
          <td>17850.0</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>1</th>
          <td>536365</td>
          <td>71053</td>
          <td>WHITE METAL LANTERN</td>
          <td>6</td>
          <td>12/1/2010 8:26</td>
          <td>3.39</td>
          <td>17850.0</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>2</th>
          <td>536365</td>
          <td>84406B</td>
          <td>CREAM CUPID HEARTS COAT HANGER</td>
          <td>8</td>
          <td>12/1/2010 8:26</td>
          <td>2.75</td>
          <td>17850.0</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>3</th>
          <td>536365</td>
          <td>84029G</td>
          <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
          <td>6</td>
          <td>12/1/2010 8:26</td>
          <td>3.39</td>
          <td>17850.0</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>4</th>
          <td>536365</td>
          <td>84029E</td>
          <td>RED WOOLLY HOTTIE WHITE HEART.</td>
          <td>6</td>
          <td>12/1/2010 8:26</td>
          <td>3.39</td>
          <td>17850.0</td>
          <td>United Kingdom</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>



Let’s clean the data to make sure there aren’t any missing values.

.. code:: python

    >>> data_df.drop(data_df.index[data_df.CustomerID != data_df.CustomerID], inplace=True)
    >>> data_df = data_df.astype({'CustomerID': int}, copy=False)
    >>> data_df.head()

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>InvoiceNo</th>
          <th>StockCode</th>
          <th>Description</th>
          <th>Quantity</th>
          <th>InvoiceDate</th>
          <th>UnitPrice</th>
          <th>CustomerID</th>
          <th>Country</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>536365</td>
          <td>85123A</td>
          <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
          <td>6</td>
          <td>2010-12-01 08:26:00</td>
          <td>2.55</td>
          <td>17850</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>1</th>
          <td>536365</td>
          <td>71053</td>
          <td>WHITE METAL LANTERN</td>
          <td>6</td>
          <td>2010-12-01 08:26:00</td>
          <td>3.39</td>
          <td>17850</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>2</th>
          <td>536365</td>
          <td>84406B</td>
          <td>CREAM CUPID HEARTS COAT HANGER</td>
          <td>8</td>
          <td>2010-12-01 08:26:00</td>
          <td>2.75</td>
          <td>17850</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>3</th>
          <td>536365</td>
          <td>84029G</td>
          <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
          <td>6</td>
          <td>2010-12-01 08:26:00</td>
          <td>3.39</td>
          <td>17850</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>4</th>
          <td>536365</td>
          <td>84029E</td>
          <td>RED WOOLLY HOTTIE WHITE HEART.</td>
          <td>6</td>
          <td>2010-12-01 08:26:00</td>
          <td>3.39</td>
          <td>17850</td>
          <td>United Kingdom</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>



Note that some of these transactions are for returns (denoted by
negative quantity values).

.. code:: python

    >>> data_df[data_df.Quantity < 1].head()


.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>InvoiceNo</th>
          <th>StockCode</th>
          <th>Description</th>
          <th>Quantity</th>
          <th>InvoiceDate</th>
          <th>UnitPrice</th>
          <th>CustomerID</th>
          <th>Country</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>141</th>
          <td>C536379</td>
          <td>D</td>
          <td>Discount</td>
          <td>-1</td>
          <td>2010-12-01 09:41:00</td>
          <td>27.50</td>
          <td>14527</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>154</th>
          <td>C536383</td>
          <td>35004C</td>
          <td>SET OF 3 COLOURED  FLYING DUCKS</td>
          <td>-1</td>
          <td>2010-12-01 09:49:00</td>
          <td>4.65</td>
          <td>15311</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>235</th>
          <td>C536391</td>
          <td>22556</td>
          <td>PLASTERS IN TIN CIRCUS PARADE</td>
          <td>-12</td>
          <td>2010-12-01 10:24:00</td>
          <td>1.65</td>
          <td>17548</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>236</th>
          <td>C536391</td>
          <td>21984</td>
          <td>PACK OF 12 PINK PAISLEY TISSUES</td>
          <td>-24</td>
          <td>2010-12-01 10:24:00</td>
          <td>0.29</td>
          <td>17548</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>237</th>
          <td>C536391</td>
          <td>21983</td>
          <td>PACK OF 12 BLUE PAISLEY TISSUES</td>
          <td>-24</td>
          <td>2010-12-01 10:24:00</td>
          <td>0.29</td>
          <td>17548</td>
          <td>United Kingdom</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>



Though customers may have returned these products, they did initially
purchase the products (which reflects an interest in the product), so
we’ll keep the intiial purchases. However, we’ll remove the return
transactions (which will also remove any discount transactions as well).

.. code:: python

    >>> data_df.drop(data_df.index[data_df.Quantity <= 0], inplace=True)
    >>> data_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>InvoiceNo</th>
          <th>StockCode</th>
          <th>Description</th>
          <th>Quantity</th>
          <th>InvoiceDate</th>
          <th>UnitPrice</th>
          <th>CustomerID</th>
          <th>Country</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>536365</td>
          <td>85123A</td>
          <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
          <td>6</td>
          <td>2010-12-01 08:26:00</td>
          <td>2.55</td>
          <td>17850</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>1</th>
          <td>536365</td>
          <td>71053</td>
          <td>WHITE METAL LANTERN</td>
          <td>6</td>
          <td>2010-12-01 08:26:00</td>
          <td>3.39</td>
          <td>17850</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>2</th>
          <td>536365</td>
          <td>84406B</td>
          <td>CREAM CUPID HEARTS COAT HANGER</td>
          <td>8</td>
          <td>2010-12-01 08:26:00</td>
          <td>2.75</td>
          <td>17850</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>3</th>
          <td>536365</td>
          <td>84029G</td>
          <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
          <td>6</td>
          <td>2010-12-01 08:26:00</td>
          <td>3.39</td>
          <td>17850</td>
          <td>United Kingdom</td>
        </tr>
        <tr>
          <th>4</th>
          <td>536365</td>
          <td>84029E</td>
          <td>RED WOOLLY HOTTIE WHITE HEART.</td>
          <td>6</td>
          <td>2010-12-01 08:26:00</td>
          <td>3.39</td>
          <td>17850</td>
          <td>United Kingdom</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>


Community Detection
===================

Let’s now find the communities of customers with similar purchases /
interests.

First, we’ll need to create a bipartite graph of customers and products.

.. code:: python

    >>> bipartite_graph = nx.from_pandas_edgelist(data_df, 'CustomerID', 'StockCode')

Next, we’ll need to use a graph projection to create a graph of
customers linked to other customers who’ve bought the same product.

.. code:: python

    >>> similarity_graph = nx.projected_graph(bipartite_graph, data_df.CustomerID.unique())

Now, we’ll need to use Louvain community detection to find customer
similarity communities.

.. code:: python

    >>> r = mg.resolver
    >>> similarity_graph_wrapped = r.wrappers.EdgeMap.NetworkXEdgeMap(similarity_graph)
    >>> labels, modularity_score = r.algos.clustering.louvain_community(similarity_graph_wrapped)

Let’s now merge the labels into our dataframe.

.. code:: python

    >>> data_df['CustomerCommunityLabel'] = data_df.CustomerID.map(lambda customer_id: labels.value[customer_id])
    >>> data_df.sample(10)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>InvoiceNo</th>
          <th>StockCode</th>
          <th>Description</th>
          <th>Quantity</th>
          <th>InvoiceDate</th>
          <th>UnitPrice</th>
          <th>CustomerID</th>
          <th>Country</th>
          <th>CustomerCommunityLabel</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>314113</th>
          <td>564562</td>
          <td>22384</td>
          <td>LUNCH BAG PINK POLKADOT</td>
          <td>10</td>
          <td>2011-08-26 09:58:00</td>
          <td>1.65</td>
          <td>15203</td>
          <td>United Kingdom</td>
          <td>2</td>
        </tr>
        <tr>
          <th>79522</th>
          <td>542996</td>
          <td>22822</td>
          <td>CREAM WALL PLANTER HEART SHAPED</td>
          <td>16</td>
          <td>2011-02-02 12:10:00</td>
          <td>4.95</td>
          <td>15046</td>
          <td>United Kingdom</td>
          <td>0</td>
        </tr>
        <tr>
          <th>412667</th>
          <td>572295</td>
          <td>23047</td>
          <td>PAPER LANTERN 5 POINT SEQUIN STAR</td>
          <td>1</td>
          <td>2011-10-23 13:49:00</td>
          <td>5.75</td>
          <td>16686</td>
          <td>United Kingdom</td>
          <td>2</td>
        </tr>
        <tr>
          <th>125657</th>
          <td>547055</td>
          <td>22090</td>
          <td>PAPER BUNTING RETROSPOT</td>
          <td>3</td>
          <td>2011-03-20 12:14:00</td>
          <td>2.95</td>
          <td>16444</td>
          <td>United Kingdom</td>
          <td>2</td>
        </tr>
        <tr>
          <th>483611</th>
          <td>577504</td>
          <td>21990</td>
          <td>MODERN FLORAL STATIONERY SET</td>
          <td>2</td>
          <td>2011-11-20 12:36:00</td>
          <td>1.25</td>
          <td>14159</td>
          <td>United Kingdom</td>
          <td>3</td>
        </tr>
        <tr>
          <th>231787</th>
          <td>557281</td>
          <td>20760</td>
          <td>GARDEN PATH POCKET BOOK</td>
          <td>2</td>
          <td>2011-06-19 12:23:00</td>
          <td>0.85</td>
          <td>13632</td>
          <td>United Kingdom</td>
          <td>0</td>
        </tr>
        <tr>
          <th>269930</th>
          <td>560534</td>
          <td>23245</td>
          <td>SET OF 3 REGENCY CAKE TINS</td>
          <td>4</td>
          <td>2011-07-19 12:10:00</td>
          <td>4.95</td>
          <td>14911</td>
          <td>EIRE</td>
          <td>3</td>
        </tr>
        <tr>
          <th>51728</th>
          <td>540672</td>
          <td>84631</td>
          <td>FRUIT TREE AND BIRDS WALL PLAQUE</td>
          <td>2</td>
          <td>2011-01-10 15:51:00</td>
          <td>7.95</td>
          <td>15281</td>
          <td>United Kingdom</td>
          <td>0</td>
        </tr>
        <tr>
          <th>218142</th>
          <td>556019</td>
          <td>23171</td>
          <td>REGENCY TEA PLATE GREEN</td>
          <td>12</td>
          <td>2011-06-08 12:02:00</td>
          <td>1.65</td>
          <td>18092</td>
          <td>United Kingdom</td>
          <td>1</td>
        </tr>
        <tr>
          <th>387217</th>
          <td>570272</td>
          <td>22300</td>
          <td>COFFEE MUG DOG + BALL DESIGN</td>
          <td>6</td>
          <td>2011-10-10 10:45:00</td>
          <td>2.55</td>
          <td>14297</td>
          <td>United Kingdom</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>



We now have clusters of customers who’ve bought similar products and can
market based on those interests.

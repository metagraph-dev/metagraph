Use Case 3: Customer Interest Clustering
========================================

Download this as a :download:`notebook </_downloads/notebooks/ecommerce.tar.gz>`.

This is a tutorial on how to perform customer clustering based on the
interests and purchases of customers.

Marketing teams frequently are interested in this analysis.

We’ll show how graph analytics can be used to gain insights about
the interests of customers by finding communities of customers who’ve
bought similar products.

We’ll accomplish this by creating a bipartite graph of customers and
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
we’ll keep the initial purchases. However, we’ll remove the return
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

    >>> customer_similarity_graph = nx.projected_graph(bipartite_graph, data_df.CustomerID.unique())

Now, we’ll need to use Louvain community detection to find similar
communities based on purchased products.

.. code:: python

    >>> r = mg.resolver
    >>> customer_similarity_graph_wrapped = r.wrappers.EdgeMap.NetworkXEdgeMap(customer_similarity_graph)
    >>> labels, modularity_score = r.algos.clustering.louvain_community(customer_similarity_graph_wrapped)

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
          <th>366736</th>
          <td>568791</td>
          <td>22179</td>
          <td>SET 10 NIGHT OWL LIGHTS</td>
          <td>2</td>
          <td>2011-09-29 09:51:00</td>
          <td>6.75</td>
          <td>16650</td>
          <td>United Kingdom</td>
          <td>3</td>
        </tr>
        <tr>
          <th>452094</th>
          <td>575322</td>
          <td>21889</td>
          <td>WOODEN BOX OF DOMINOES</td>
          <td>18</td>
          <td>2011-11-09 13:36:00</td>
          <td>1.25</td>
          <td>14543</td>
          <td>United Kingdom</td>
          <td>2</td>
        </tr>
        <tr>
          <th>99049</th>
          <td>544692</td>
          <td>84406B</td>
          <td>CREAM CUPID HEARTS COAT HANGER</td>
          <td>4</td>
          <td>2011-02-23 09:14:00</td>
          <td>4.15</td>
          <td>17499</td>
          <td>United Kingdom</td>
          <td>3</td>
        </tr>
        <tr>
          <th>485779</th>
          <td>577669</td>
          <td>22762</td>
          <td>CUPBOARD 3 DRAWER MA CAMPAGNE</td>
          <td>1</td>
          <td>2011-11-21 10:48:00</td>
          <td>14.95</td>
          <td>15567</td>
          <td>United Kingdom</td>
          <td>0</td>
        </tr>
        <tr>
          <th>315010</th>
          <td>564715</td>
          <td>22030</td>
          <td>SWALLOWS GREETING CARD</td>
          <td>12</td>
          <td>2011-08-28 11:01:00</td>
          <td>0.42</td>
          <td>14472</td>
          <td>United Kingdom</td>
          <td>0</td>
        </tr>
        <tr>
          <th>230147</th>
          <td>557123</td>
          <td>22551</td>
          <td>PLASTERS IN TIN SPACEBOY</td>
          <td>4</td>
          <td>2011-06-16 17:56:00</td>
          <td>1.65</td>
          <td>15555</td>
          <td>United Kingdom</td>
          <td>3</td>
        </tr>
        <tr>
          <th>237143</th>
          <td>557802</td>
          <td>21677</td>
          <td>HEARTS  STICKERS</td>
          <td>12</td>
          <td>2011-06-23 10:15:00</td>
          <td>0.85</td>
          <td>15130</td>
          <td>United Kingdom</td>
          <td>0</td>
        </tr>
        <tr>
          <th>308387</th>
          <td>563952</td>
          <td>82494L</td>
          <td>WOODEN FRAME ANTIQUE WHITE</td>
          <td>8</td>
          <td>2011-08-22 10:44:00</td>
          <td>2.95</td>
          <td>15572</td>
          <td>United Kingdom</td>
          <td>0</td>
        </tr>
        <tr>
          <th>285501</th>
          <td>561910</td>
          <td>21843</td>
          <td>RED RETROSPOT CAKE STAND</td>
          <td>6</td>
          <td>2011-08-01 10:23:00</td>
          <td>9.95</td>
          <td>17053</td>
          <td>United Kingdom</td>
          <td>0</td>
        </tr>
        <tr>
          <th>294263</th>
          <td>562708</td>
          <td>21621</td>
          <td>VINTAGE UNION JACK BUNTING</td>
          <td>3</td>
          <td>2011-08-08 14:15:00</td>
          <td>8.50</td>
          <td>17085</td>
          <td>United Kingdom</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>



We now have clusters of customers who’ve bought similar products and can
market to these interests. 

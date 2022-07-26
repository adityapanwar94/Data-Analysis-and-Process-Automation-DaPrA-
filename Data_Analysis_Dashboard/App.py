"""Data-Analysis-and-Process-Automation Dashboard

This script creates a fully functional webpage for automating data
analysis and clustering processes on any unlabelled dataset. 
It is assumed that the dataset uploaded has been preprocessed 
for null values and appropriately cleaned.

This tool accepts dataset in csv format.

This script requires modules mentioned in requrements.txt to be installed 
within the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * main - the main function of the script
"""
#========================================================================================
# Librabries Imported
import copy
import string
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
from sklearn.metrics import silhouette_samples, silhouette_score
from plotly.subplots import make_subplots
#========================================================================================

def main():  
    """ The functions contains the complete script for the webpage """
    #-------------------------------------------------------------------------------
    # Title and About Section
    st.title("Data-Analysis-and-Process-Automation Dashboard")
    image = Image.open(r'Data_Analysis_Dashboard\img1.jpg')
    st.image(image, caption='Check About', use_column_width='always')
    with st.expander("About the webpage"):
        st.write("""
            This dashboard allows us to analyse the real-life unlabelled dataset from the eyantra registration portal. 
            -   You will find the following functionalities:
                -   View dataset in tabular format
                -   Drop columns based on your need
                -   Add description for dataset columns and represent it over the webpage
                -   Automated data analysis chart with interactive front-end
                -   Data clustering visualisation
        """)
    st.sidebar.title("Data Analsis Controller")
    #-------------------------------------------------------------------------------
    # Loading the dataset...
    # @st.cache(allow_output_mutation=True)
    def load_data():
        """ The functions contains the script to load the data. 
        Note: st.cache creates a cache copy of the uploaded dataset"""

        data = pd.read_csv("Data_Analysis_Dashboard\Refine_data(07-07-22).csv")
        data = data.drop(['Unnamed: 0'], axis=1)
        return data
    dataset = load_data()
    # Representation of tabular dataset on the webpage...
    if st.sidebar.checkbox("Show Dataset",False,key=0):
        st.markdown("## Dataset in Tabular Format")
        st.markdown("### The Pre-Processed Dataset")
        st.write(dataset)
    #-------------------------------------------------------------------------------
    # Variable storing column list...
    collist = dataset.columns.to_list()

    # Dropping columns from the dataset on the webpage...
    if st.sidebar.checkbox("Drop Columns",False,key=1):
        st.markdown("### Drop Irrelevant Columns")
        with st.expander("Guide to this feature"):
            st.write("""
                This feature allows us to control the dimensions of the dataset by dropping user-selected columns. 
                -   **How to use:**
                    -   Select the column names from the dropdown.
                    -   As you perform the selection you will see the change in the dimensions of the dataset immediately, after each selection.
                    -   To revert back to original dataset select the checkbox given below
            """)
        options = st.multiselect('Select the Columns to be dropped', collist)
        dataset_updated = dataset.drop(options,axis=1)
        collist = dataset_updated.columns.to_list()
        # To revert to original checkbox...
        error = st.checkbox('To revert back to original select this', value=False)
        if error:
            dataset_updated = dataset
            collist = dataset_updated.columns.to_list()
        # Final dataset's dimension representation on webpage...
        st.write(dataset_updated.shape)
        st.write(collist)
    else:
        dataset_updated = dataset
   #-------------------------------------------------------------------------------
    # Add Column Information on the webpage...
    if st.sidebar.checkbox("Add Feature Info",False,key=2):
        st.markdown("### • Generate Column Description Table")
        with st.expander("Guide to this feature"):
            st.write("""
                **To add description for each column follow the steps:**
                -   Select the column names from the dropdown.
                -   As you select, a form field for each column will be generated below.
                -   Fill the form with all the details and submit.
                -   A list will be generated below for future reference too.
            """)
        # Multiselect option for column selection...
        options = st.multiselect('Select the columns for which information has to be provided', collist)
        labels = copy.deepcopy(options)
        names = options
        # Form fields generated for selected columns...
        with st.form(key='info'):
            for i,name in enumerate(names):
                names[i] = st.text_input(names[i], key=i)
            submit_button = st.form_submit_button(label='submit')
        # User entered data represented after submission...    
        # if submit_button:
        for i, name in enumerate(names):
            cols = st.columns(2)
            cols[0].write(labels[i])
            cols[1].write("{}".format(name))
    #-------------------------------------------------------------------------------
    # Visual data analysis begins...            
    st.markdown("## The Visual Data Analysis 📈")
    #*************************************************************
    # Automated Bar plots for webpage...
    st.markdown("### • Bar Plots Diagram")
    with st.expander("Guide to this feature"):
        st.write("""
            This feature generates interactive and flexible bar plots. 
            -   **How to use:**
                -   Select column name/s from the dropdown.
                -   As you perform the selection you will see appearance of bar plots with legends.
        """)
    optionsb = st.multiselect('Select columns for analysis', collist, key=5)
    labelsb = copy.deepcopy(optionsb)
    if len(optionsb) > 0:
        datfrm = pd.DataFrame([])
        columns = optionsb
        for cols in columns:
            temp =  pd.DataFrame([])
            Count_analysis = pd.DataFrame(dataset_updated[cols].value_counts().sort_index())
            temp = pd.concat([datfrm,Count_analysis], axis = 1)
            datfrm = temp
        fig = px.bar(datfrm, barmode='group', height=400, width=800)
        st.write(fig)
    else:
        st.write("Select columns first")
    #****************************************************************
    # Automated Correlation Plots between two columns for the webpage...
    st.markdown("### • Correlation Plots Diagram")
    with st.expander("Guide to this feature"):
        st.write("""
            This feature generates interactive and flexible correlation plots. 
            -   **How to use:**
                -   Select only 2 column from the dropdown.
                -   As you perform the selection you will see appearance of correlation plot with legends.
        """)
    optionss = st.multiselect('Select two columns for analysis', collist, key=6)
    labelss = copy.deepcopy(optionss)
    if len(optionss) == 2:
        fig = px.scatter(dataset_updated, x=optionss[0], y=optionss[1])
        st.write(fig)
        corr1 = dataset_updated.iloc[:,0].corr(dataset_updated.iloc[:,1],method='pearson', min_periods=3)
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Pearson Correlation Value of the features-**")
        with col_right:
            st.write(corr1)
    else:
        st.write("Select columns first") 
    #***************************************************************
    # Automated Parallel Category Diagrams for the webpage...
    if st.sidebar.checkbox("Parallel Diagram",False,key=7):
        st.markdown("### • Parallel Category Diagram")
        with st.expander("Guide to this feature"):
            st.write("""
                **About Parallel Category Diagram:**
                -   The parallel categories diagram is a visualization of multi-dimensional categorical data sets.
                -   Each variable in the data set is represented by a column of rectangles, where each rectangle corresponds to a discrete value taken on by that variable. 
                -   The relative heights of the rectangles reflect the relative frequency of occurrence of the corresponding value.
                -   Combinations of category rectangles across dimensions are connected by ribbons, where the height of the ribbon corresponds to the relative frequency of occurrence of the combination of categories in the data set.
                **How to use:**
                -   Select 4 column from the dropdown.
                -   After the selection you will see appearance of parallel plots with legends.
                -   Use the "Select colour option" dropdown to define the colour scheme.
            """)
        colorlist = ['Plotly3', 'dense', 'Turbo', 'Rainbow', 'Burg', 'Sunsetdark', 'Agsunset']
        options = st.multiselect('Select 4 columns for analysis', collist, key=8)
        coloroptions = st.selectbox('Select a colour option', colorlist )
        labels = copy.deepcopy(options)
        if len(options) == 4:
            col1 = go.parcats.Dimension(
                values = dataset_updated[options[0]].values,
                label = labels[0],
                categoryarray = pd.unique(dataset_updated[options[0]]).tolist(),
                categoryorder= 'category ascending',
                # ticktext= ['edible','poisonous']

            )
            col2 = go.parcats.Dimension(
                values = dataset_updated[options[1]].values,
                label = labels[1],
                categoryarray = pd.unique(dataset_updated[options[1]]).tolist(),
                #categoryorder= 'category ascending',
                # ticktext= ['Bell','conical','flat','knobbed','sunken','convex'],

            )
            col3 = go.parcats.Dimension(
                values = dataset_updated[options[2]].values,
                label = labels[2],
                categoryarray = pd.unique(dataset_updated[options[2]]).tolist(),
                #categoryorder= 'category ascending',
                # ticktext= ['fibrous','grooves','smooth','scaly']

            )
            col4 = go.parcats.Dimension(
                values = dataset_updated[options[3]].values,
                label = labels[3],
                categoryarray = pd.unique(dataset_updated[options[3]]).tolist(),
                #categoryorder= 'category ascending',
                # ticktext= ['buff','cinnamon','red','gray','brown','pink','green','purple','white','yellow']

            )
            color = dataset_updated[options[0]].values
            fig = go.Figure(data = [go.Parcats(dimensions=[col1, col2, col3, col4],
            line={'color': color, 'colorscale': coloroptions},
            hoveron='dimension', hoverinfo='count+probability',
            labelfont={'size': 18, 'family': 'Times'},
            tickfont={'size': 16, 'family': 'Times'},
            arrangement='fixed')])
            fig.update_layout(height=500, width=800,paper_bgcolor='rgb(243, 243, 243)')
            st.write(fig)
        else:
            st.write("Select 4 columns first ")
    #-------------------------------------------------------------------------------
    # K-Means Clustering Begins...
    st.markdown("## **K-Means Clustering**")
    with st.expander("This section contains..."):
            st.write("""
                -   **K-Means Controller on the sidebar.**
                    -   The controller provides all the options and functionalities to implement K-means clustering such as:
                        -   **Normalise Dataset**
                            -   Select the checkbox on sidebar to scale the datapoints within the range of 0 to 1 (This is highly advisable for PCA implementation).
                            -   You can also view the scaled dataset through the sidebar option.
                        -   **Feature Weights**
                        -   **K-Means with PCA**
                        -   **K-Means without PCA**
            """)
    #---------------------------------------------------------------------------------
    # Controller for K-Means functionality...
    st.sidebar.title("K-Means Controller")
    #-------------------------------------------------------------------------------
    # Checkbox to allow Normalisation...
    scale = st.sidebar.checkbox('Normalise dataset(It is advisable to do so)', value=False)
    if scale:
        Grouped_dataC = dataset_updated.copy()
        scaler = MinMaxScaler()
        data_columns = Grouped_dataC.drop(['team_id'], axis=1).columns
        data_scaled = pd.DataFrame(scaler.fit_transform(Grouped_dataC.drop(['team_id'], axis=1)), columns=data_columns)
        # Checkbox to view scaled dataset on the webpage...
        view_data = st.sidebar.checkbox('To view scaled dataset', value=False)
        if view_data:
            st.write(data_scaled)
    else:
        data_scaled = dataset_updated
    #---------------------------------------------------------------------------------
    # Checkbox to allow Weight Defining...
    weight = st.sidebar.checkbox('Feature Weights', value=False, key=9)
    w_flag = 0
    if weight:
        st.write("### **• Setting up feature weights**")
        with st.expander("About this feature..."):
            st.write("""
                **How to use:**
                -   Select the columns for which you want to define weights from the form's dropdown or select to checkbox to choose all columns at once.
                -   Set up the max limit of the weights to be assigned.
                -   Set up the min limit of the weights to be assigned. (This is done to set max and min range of sliders)
                -   Submit the form to create weight slider for each feature.
                -   Set up slider value for each feature column and hit submit.
                -   To view the weighted dataset, select the checkbox. 
            """)
        # Forms to select columns, max values, min values...
        with st.form("Weight_form"):
            collist_w = data_scaled.columns.to_list()
            wcolptions = st.multiselect('Select the columns for defining weights', collist_w)
            min_val = float(st.number_input("Set up min value", value=0,key=10))
            max_val = float(st.number_input("Set up max value", value=1000,key=11))
            w_allcols = st.checkbox('For selecting all columns', value=False, key=12) 
            submit = st.form_submit_button("Submit")
        # When all columns are selected...    
        if w_allcols:
            col_for_form = collist_w
            labels = copy.deepcopy(col_for_form)
            with st.form(key='weight_input'):
                for i,col in enumerate(col_for_form):
                    col_for_form[i] = st.slider(labels[i], min_value=min_val, max_value=max_val, value=10.0, step=1.0)
                submit2 = st.form_submit_button(label='Submit')
            for i in range(0,len(labels)):
                data_scaled[labels[i]] = data_scaled[labels[i]].apply(lambda x: x*col_for_form[i])
                # Checkbox to view scaled dataset on the webpage...
            view_dataweights = st.checkbox('To view scaled dataset', value=False, key=13)
            if view_dataweights:
                st.write(data_scaled)
        # For specific column selected...
        else:
            col_for_form = wcolptions
            labels = copy.deepcopy(col_for_form)
            # Form to enter weights for features...
            with st.form(key='weight_input'):
                for i,col in enumerate(col_for_form):
                    col_for_form[i] = st.slider(labels[i], min_value=min_val, max_value=max_val, value=10.0, step=0.5)
                submit2 = st.form_submit_button(label='Submit')
            # Implementing weight multiplication...
            for i in range(0,len(labels)):
                data_scaled[labels[i]] = data_scaled[labels[i]].apply(lambda x: x*col_for_form[i])
            # Checkbox to view scaled dataset on the webpage...
            view_dataweights = st.checkbox('To view scaled dataset', value=False, key=13)
            if view_dataweights:
                st.write(data_scaled)
    #---------------------------------------------------------------------------------
    # K-Means with Principal Component Analysis
    if st.sidebar.checkbox("K-Means with PCA",False,key=14):
        flag = 0
        st.write("### **• K-Means with Principal Component Analysis**")
        with st.expander("About this section..."):
            st.write("""
                **About PCA and its features:**
                -   In short, PCA is a dimensionality reduction technique that transforms a set of features in a dataset into a smaller number of features called principal components while at the same time trying to retain as much information in the original dataset as possible:
            """)
            col1, col2, col3 = st.columns([1,6,1])
            with col1:
                st.write("")
            with col2:
                image1 = Image.open(r'Data_Analysis_Dashboard\pca.png')
                st.image(image1, caption='Check About', width=500)
            with col3:
                st.write("")
            st.write("""
                **Parameters used:**
                -   n_components- int, float, default=None
                    -   Number of components to keep. if n_components is not set all components are kept
                -   svd_solver- {'auto', 'full', 'arpack', 'randomized'}, default='auto'
            """)
            url = r"https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"
            st.markdown("check out this [link](%s) to know more about the features of PCA" % url)

            url1 = r"https://towardsdatascience.com/using-principal-component-analysis-pca-for-machine-learning-b6e803f5bf1e"
            st.markdown("To know more about PCA implemententation, check this [link](%s)" % url1)      
        column_left, column_right = st.columns(2)
        #_________________________________________________________________________
        with column_left:
            with st.expander("Guide to PCA Forms"):
                st.write("""
                    **Follow the steps:**
                    -   Either set the number of components or variance value for the PCA implementation. (make sure the other one is zero)
                    -   Define the type of svd_solver.(Auto is recommended)
                    -   Set the maximum number of clusters to be analysed under elbow graph.
                """)  
            # Form creation for users to specify pca parameters and maximum clusters...
            with st.form("PCA_form"):
                components1 = int(st.number_input("Number of components", value=0,key=15))
                components2 = st.number_input("Variance value", value=0.0,key=16)
                Svd_Solver = st.selectbox('Svd_Solver', ('auto', 'full', 'arpack', 'randomized'))
                max_clusters = int(st.number_input("Maximum number of Clusters", value=0,key=17))
                submit = st.form_submit_button("Submit")
        # PCA when variance percentage is specified...
        if components2 != 0.0:
            #__________________________________________________________________________
            # PCA Implementation with user-defined parameters...
            pcacolname = []
            pca = PCA(n_components=components2, svd_solver=Svd_Solver)
            principalComponents = pca.fit_transform(data_scaled)
            principalDf = pd.DataFrame(data = principalComponents)
            flag = 1
            #__________________________________________________________________________
            # Checkbox to view PCA generated values...
            view_pcadet = st.checkbox('To view the variance and PCA reduced dataset', value=False)
            pcacol_left, pcacol_right = st.columns(2)
            if view_pcadet:
                with pcacol_left:
                    st.write("### **• PCA's Explained Variance Ratio**")
                    st.write(pca.explained_variance_ratio_.cumsum() * 100)
                with pcacol_right:
                    st.write("### **• PCA's Dimensionality Reduced Dataset**")
                    st.write(principalDf)
                st.write('### **• Top 4 most important features in each component**')
                pca_components = abs(pca.components_)
                for row in range(pca_components.shape[0]):
                    # get the indices of the top 4 values in each row
                    temp = np.argpartition(-(pca_components[row]), 4) 
                    # sort the indices in descending order
                    indices = temp[np.argsort((-pca_components[row])[temp])][:4]
                    # print the top 4 feature names
                    st.write(f'Component {row}: {data_scaled.columns[indices].to_list()}')
                # Checkbox to view PCA's visual Analysis...
                if principalDf.shape[1] <= 20:
                    st.write("### **• PCA Visualisation**")
                    labels = {str(i): f"PC {i+1} ({var:.1f}%)"
                    for i, var in enumerate(pca.explained_variance_ratio_ * 100)}
                    fig = px.scatter_matrix(
                        principalComponents,
                        labels=labels,
                        dimensions=range(len(pca.explained_variance_ratio_)))
                    fig.update_traces(diagonal_visible=False)
                    st.write(fig)
    
        # PCA when component number is specified...
        else:
            pcacolname = []
            az_Upper = string.ascii_uppercase
            for i in range(components1):
                pcacolname.append(az_Upper[i])
            #_________________________________________________________________________
            with column_right:
                with st.expander("Guide to column form"):
                    st.write("""
                        **Follow the steps:**
                        -   If you have defined the number of components, then you have to label each component column through the form.
                        -   Fill up the column names in the form given below and hit submit.
                    """)  
                # Form creation for users to define column names for PCA generated dataset...
                with st.form(key='columnname'):
                    for i,pcacol in enumerate(pcacolname):
                        pcacolname[i] = st.text_input("component-"+pcacolname[i], key = i)
                    submitcol = st.form_submit_button(label='Submit Names')
            #__________________________________________________________________________
            # PCA Implementation with user-defined parameters...
            if len(pcacolname) != 0:
                if pcacolname[0] != "":
                    pca = PCA(n_components=components1, svd_solver=Svd_Solver)
                    principalComponents = pca.fit_transform(data_scaled)
                    principalDf = pd.DataFrame(data = principalComponents, columns=pcacolname)
                    flag = 1
                    # Checkbox to view PCA generated values...
                    view_pcadet = st.checkbox('To view the variance and PCA reduced dataset', value=False)
                    pcacol_left, pcacol_right = st.columns(2)
                    if view_pcadet:
                        with pcacol_left:
                            st.write("### **• PCA's Explained Variance Ratio**")
                            st.write(pca.explained_variance_ratio_.cumsum() * 100)
                        with pcacol_right:
                            st.write("### **• PCA's Dimensionality Reduced Dataset**")
                            st.write(principalDf)
                        # Checkbox to view PCA's visual Analysis...
                        st.write("### **• PCA Visualisation**")
                        labels = {str(i): f"PC {i+1} ({var:.1f}%)"
                        for i, var in enumerate(pca.explained_variance_ratio_ * 100)}
                        fig = px.scatter_matrix(
                            principalComponents,
                            labels=labels,
                            dimensions=range(len(pca.explained_variance_ratio_)))
                        fig.update_traces(diagonal_visible=False)
                        st.write(fig)

                        st.write('### **• Top 4 most important features in each component**')
                        pca_components = abs(pca.components_)
                        for row in range(pca_components.shape[0]):
                            # get the indices of the top 4 values in each row
                            temp = np.argpartition(-(pca_components[row]), 4) 
                            # sort the indices in descending order
                            indices = temp[np.argsort((-pca_components[row])[temp])][:4]
                            # print the top 4 feature names
                            st.write(f'Component {row}: {data_scaled.columns[indices].to_list()}')
                    view_kmleft, view_kmright = st.columns(2)
        #__________________________________________________________________________
        # K-Means on PCA reduced dataset...
        if flag == 1:
            wcss = []
            K = range(1,max_clusters)
            for k in K:
                km = KMeans(n_clusters=k)
                km = km.fit(principalDf)
                wcss.append(km.inertia_)
            st.write("### **• The Elbow Graph**")
            with st.expander("About K-Means Clustering"):
                st.write("""
                    -   **Introduction**
                        -   Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group.
                    -   **Elbow Method**
                        -   Elbow method gives us an idea on what a good k number of clusters would be based on the sum of squared distance (SSE) between data points and their assigned clusters’ centroids. We pick k at the spot where SSE starts to flatten out and forming an elbow.
                    -   **Silhouette Analysis**
                        -   Silhouette analysis can be used to determine the degree of separation between clusters. 
                """)
                url2 = r"https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a#:~:text=Kmeans%20clustering%20is%20one%20of,into%20distinct%20non%2Doverlapping%20subgroups."
                st.markdown("To know more about K-Means, check this [link](%s)" % url2) 

            fig = go.Figure(data = go.Scatter(x = np.array(K), y = wcss))
            fig.update_layout(title='WCSS vs. Cluster number', xaxis_title='Clusters', yaxis_title='WCSS')
            st.write(fig)
            # Input box to select the optimal number of clusters from elbow graph...
            opt_cluster = int(st.number_input("Select the optimal cluster value from elbow graph",value=0,key=18))
            
            if opt_cluster:
                algorithm_pca = (KMeans(n_clusters = opt_cluster ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 100  , algorithm='elkan') )
                algorithm_pca.fit(principalDf)
                labels_pca = algorithm_pca.labels_
                centroids_pca = algorithm_pca.cluster_centers_
                # Checkbox to view centroid and labels developed from clusters...
                view_kmdet = st.checkbox("To view the centroid and the labels", value=False)
                col_lab, col_center = st.columns(2)
                if view_kmdet:
                    with col_lab:
                        st.write("### **• Cluster Labels**")
                        st.write(np.unique(labels_pca))
                    with col_center:
                        st.write("### **• Cluster Centroid**")
                        st.write(centroids_pca)
                    
                # Visualise 2-Dimensional and 3-Dimensional clustering...
                if len(pcacolname) == 0:
                    # Scatter plot for two-dimensional PCA datset... 
                    if principalDf.shape[1] == 2:
                        principalDf.columns = ['col1', 'col2']
                        fig = px.scatter(principalDf, x='col1', y = 'col2', color=labels_pca)
                        st.write(fig)
                     # 3-D Scatter plot for three-dimensional PCA datset...
                    elif principalDf.shape[1] == 3:
                        principalDf.columns = ['col1', 'col2', 'col3']  
                        fig = px.scatter_3d(principalDf, x = 'col1', y='col2', z='col3',
                        color=labels_pca, opacity = 0.8, size_max=30)
                        st.write(fig)
                else:
                    # Scatter plot for two-dimensional PCA datset...    
                    if len(pcacolname) == 2:
                        fig = px.scatter(principalDf, x=pcacolname[0], y = pcacolname[1], color=labels_pca)
                        st.write(fig)
                    # 3-D Scatter plot for three-dimensional PCA datset...  
                    if len(pcacolname) == 3:
                        fig = px.scatter_3d(principalDf, x = pcacolname[0], y=pcacolname[1], z=pcacolname[2],
                        color=labels_pca, opacity = 0.8, size_max=30)
                        st.write(fig)
                # Visualise Sihouette Plots...
                silhouette_vals = silhouette_samples(principalDf, labels_pca)
                fig, ax1 = plt.subplots(1)
                fig.set_size_inches(10, 7)
                centroids = algorithm_pca.cluster_centers_
                y_ticks = []
                y_lower, y_upper = 0, 0
                for i, cluster in enumerate(np.unique(labels_pca)):
                    cluster_silhouette_vals = silhouette_vals[labels_pca == cluster]
                    cluster_silhouette_vals.sort()
                    y_upper += len(cluster_silhouette_vals)
                    ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
                    ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
                    y_lower += len(cluster_silhouette_vals)

                # Get the average silhouette score and plot it...
                avg_score = np.mean(silhouette_vals)
                ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
                ax1.set_yticks([])
                ax1.set_xlim([-0.1, 1])
                ax1.set_xlabel('Silhouette coefficient values')
                ax1.set_ylabel('Cluster labels')
                ax1.set_title('Silhouette plot for the various clusters', y=1.02)
                st.write(fig)
                # Get the team data for a particular cluster value...
                new_frame = pd.DataFrame(dataset_updated)
                new_frame['clusters'] = algorithm_pca.predict(principalDf)
                columns_frame = new_frame.columns.to_list()
                cluster_option = st.selectbox('Select the cluster',list(np.unique(labels_pca)))
                col_select = st.multiselect('Select the columns for preview',columns_frame)
                prev_data = pd.DataFrame(new_frame.loc[(new_frame['clusters'] == cluster_option),col_select])
                col_data, col_analysis = st.columns(2)
                with col_data:
                    st.write("### **• The Cluster Data**")
                    st.write(prev_data)
                with col_analysis:
                    if not prev_data.empty:
                        st.write("### **• The Cluster Data Analysis**")
                        st.write(prev_data.describe())

    #---------------------------------------------------------------------------------
    # K-Means without Principal Component Analysis...
    if st.sidebar.checkbox("K-Means without PCA",False,key=19):
        st.write("### **• K-Means without Principal Component Analysis**")
        #_______________________________________________________________________________
        # Form creation for users to specify maximum clusters and columns to include...
        with st.form("PCA2_form"):
            collistn = data_scaled.columns.to_list()
            kmcolptions = st.multiselect('Select the columns for K-means', collistn)
            allcols = int(st.number_input("To select all cols type 1 else 0", value=0,key=20))
            max_clusters = int(st.number_input("Maximum number of Clusters", value=0,key=21))
            submit = st.form_submit_button("Submit")
        if allcols == 1:
            new_dataset = data_scaled
        else:
            new_dataset = data_scaled[kmcolptions]
        #_______________________________________________________________________________
        # K-Means on User-Selected dataset...
        wcss = []
        K = range(1,max_clusters)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(new_dataset)
            wcss.append(km.inertia_)
        st.write("### **• The Elbow Graph**")
        fig = go.Figure(data = go.Scatter(x = np.array(K), y = wcss))
        fig.update_layout(title='WCSS vs. Cluster number', xaxis_title='Clusters', yaxis_title='WCSS')
        st.write(fig)
        # Input box to select the optimal number of clusters from elbow graph...
        opt_cluster = int(st.number_input("Select the optimal cluster value from elbow graph",value=0,key=22))
        if opt_cluster:
            algorithm_pca = (KMeans(n_clusters = opt_cluster ,init='k-means++', n_init = 10 ,max_iter=300, 
                    tol=0.0001,  random_state= 100  , algorithm='elkan') )
            algorithm_pca.fit(new_dataset)
            labels_pca = algorithm_pca.labels_
            centroids_pca = algorithm_pca.cluster_centers_
            # Visualise Sihouette Plots...
            silhouette_vals = silhouette_samples(new_dataset, labels_pca)
            fig, ax1 = plt.subplots(1)
            fig.set_size_inches(10, 7)
            centroids = algorithm_pca.cluster_centers_
            y_ticks = []
            y_lower, y_upper = 0, 0
            for i, cluster in enumerate(np.unique(labels_pca)):
                cluster_silhouette_vals = silhouette_vals[labels_pca == cluster]
                cluster_silhouette_vals.sort()
                y_upper += len(cluster_silhouette_vals)
                ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
                ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
                y_lower += len(cluster_silhouette_vals)

            # Get the average silhouette score and plot it
            avg_score = np.mean(silhouette_vals)
            ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
            ax1.set_yticks([])
            ax1.set_xlim([-0.1, 1])
            ax1.set_xlabel('Silhouette coefficient values')
            ax1.set_ylabel('Cluster labels')
            ax1.set_title('Silhouette plot for the various clusters', y=1.02)
            st.write(fig)
            # Checkbox to view centroid and labels developed from clusters...
            view_kmdet = st.checkbox("To view the centroid and the labels", value=False, key=23)
            col1_lab, col1_center = st.columns(2)
            if view_kmdet:
                with col1_lab:
                    st.write("### **• Cluster Labels**")
                    st.write(np.unique(labels_pca))
                with col1_center:
                    st.write("### **• Cluster Centroid**")
                    st.write(centroids_pca)
                
            # Scatter plot for two-dimensional PCA datset... 
            if len(kmcolptions) == 2:
                fig = px.scatter(new_dataset, x=kmcolptions[0], y = kmcolptions[1], color=labels_pca)
                st.write(fig)
            # 3-D Scatter plot for three-dimensional PCA datset...  
            if len(kmcolptions) == 3:
                fig = px.scatter_3d(new_dataset, x = kmcolptions[0], y=kmcolptions[1], z=kmcolptions[2],
                color=labels_pca, opacity = 0.8, size_max=30)
                st.write(fig)
            # Get the team data for a particular cluster value...
            new_frame = pd.DataFrame(dataset_updated)
            new_frame['clusters'] = algorithm_pca.predict(new_dataset)
            columns_frame = new_frame.columns.to_list()
            cluster_option = st.selectbox('Select the cluster',list(np.unique(labels_pca)))
            col_select = st.multiselect('Select the columns for preview',columns_frame)
            prev_data = pd.DataFrame(new_frame.loc[(new_frame['clusters'] == cluster_option),col_select])
            col_data, col_analysis = st.columns(2)
            with col_data:
                st.write("### **• The Cluster Data**")
                st.write(prev_data)
            with col_analysis:
                if not prev_data.empty:
                    st.write("### **• The Cluster Data Analysis**")
                    st.write(prev_data.describe())

    



if __name__ == "__main__":
    main()   




        

    




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
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#========================================================================================

def main():    
    """ The functions contains the complete script for the webpage """
    #-------------------------------------------------------------------------------
    # Title and About Section
    st.title("Data-Analysis-and-Process-Automation Dashboard")
    image = Image.open('img1.jpg')
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
    @st.cache(allow_output_mutation=True)
    def load_data():
        """ The functions contains the script to load the data. 
        Note: st.cache creates a cache copy of the uploaded dataset"""

        data = pd.read_csv("Refine_data(07-07-22).csv")
        data = data.drop(['Unnamed: 0'], axis=1)
        return data
    dataset = load_data()
    # Representation of tabular dataset on the webpage...
    if st.sidebar.checkbox("Show Dataset",False,key=5):
        st.markdown("## Dataset in Tabular Format")
        st.markdown("### The Pre-Processed Dataset")
        st.write(dataset)
    #-------------------------------------------------------------------------------
    # Variable storing column list...
    collist = dataset.columns.to_list()

    # Dropping columns from the dataset on the webpage...
    if st.sidebar.checkbox("Drop Columns",False,key=0):
        st.markdown("### Drop Irrelevant Columns")
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
    if st.sidebar.checkbox("Add Feature Info",False,key=1):
        st.markdown("### • Generate Column Description Table")
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
        if submit_button:
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
    optionsb = st.multiselect('Select columns for analysis', collist, key=0)
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
    optionss = st.multiselect('Select two columns for analysis', collist, key=1)
    labelss = copy.deepcopy(optionss)
    if len(optionss) == 2:
        fig = px.scatter(dataset_updated, x=optionss[0], y=optionss[1])
        st.write(fig)
    else:
        st.write("Select columns first") 
    #***************************************************************
    # Automated Parallel Category Diagrams for the webpage...
    if st.sidebar.checkbox("Parallel Diagram",False,key=2):
        st.markdown("### • Parallel Category Diagram")
        colorlist = ['Plotly3', 'dense', 'Turbo', 'Rainbow', 'Burg', 'Sunsetdark', 'Agsunset']
        options = st.multiselect('Select 4 columns for analysis', collist, key=3)
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
    #-------------------------------------------------------------------------------
    # Checkbox to allow Normalisation...
    scale = st.checkbox('Normalise dataset(it is advisable to do so)', value=False)
    if scale:
        Grouped_dataC = dataset_updated.copy()
        scaler = MinMaxScaler()
        data_columns = Grouped_dataC.drop(['team_id'], axis=1).columns
        data_scaled = pd.DataFrame(scaler.fit_transform(Grouped_dataC.drop(['team_id'], axis=1)), columns=data_columns)
        # Checkbox to view scaled dataset on the webpage...
        view_data = st.checkbox('To view scaled dataset', value=False)
        if view_data:
            st.write(data_scaled)
    else:
        data_scaled = dataset_updated
    #---------------------------------------------------------------------------------
    # Controller for K-Means functionality...
    st.sidebar.title("K-Means Controller")
    #---------------------------------------------------------------------------------
    # K-Means with Principal Component Analysis
    if st.sidebar.checkbox("K-Means with PCA",False,key=3):
        st.write("### **• K-Means with Principal Component Analysis**")
        column_left, column_right = st.columns(2)
        #_________________________________________________________________________
        with column_left:
            # Form creation for users to specify pca parameters and maximum clusters...
            with st.form("PCA_form"):
                components = int(st.number_input("Number of components", value=0,key=0))
                Svd_Solver = st.selectbox('Svd_Solver', ('auto', 'full', 'arpack', 'randomized'))
                max_clusters = int(st.number_input("Maximum number of Clusters", value=0,key=1))
                submit = st.form_submit_button("Submit")
        pcacolname = []
        az_Upper = string.ascii_uppercase
        for i in range(components):
            pcacolname.append(az_Upper[i])
        #_________________________________________________________________________
        with column_right:
            # Form creation for users to define column names for PCA generated dataset...
            with st.form(key='columnname'):
                for i,pcacol in enumerate(pcacolname):
                    pcacolname[i] = st.text_input("column-"+pcacolname[i], key = i)
                submitcol = st.form_submit_button(label='Submit Names')
        #__________________________________________________________________________
        # PCA Implementation with user-defined parameters...
        if len(pcacolname) != 0:
            if pcacolname[0] != "":
                pca = PCA(n_components=components, svd_solver=Svd_Solver)
                principalComponents = pca.fit_transform(data_scaled)
                principalDf = pd.DataFrame(data = principalComponents, columns=pcacolname)
                # Checkbox to view PCA generated values...
                view_pcadet = st.checkbox('To view the variance and PCA reduced dataset', value=False)
                pcacol_left, pcacol_right = st.columns(2)
                if view_pcadet:
                    with pcacol_left:
                        st.write("### **• PCA's Explained Variance Ratio**")
                        st.write(pca.explained_variance_ratio_)
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
                view_kmleft, view_kmright = st.columns(2)
                #__________________________________________________________________________
                # K-Means on PCA reduced dataset...
                wcss = []
                K = range(1,max_clusters)
                for k in K:
                    km = KMeans(n_clusters=k)
                    km = km.fit(principalDf)
                    wcss.append(km.inertia_)
                st.write("### **• The Elbow Graph**")
                fig = go.Figure(data = go.Scatter(x = np.array(K), y = wcss))
                fig.update_layout(title='WCSS vs. Cluster number', xaxis_title='Clusters', yaxis_title='WCSS')
                st.write(fig)
                # Input box to select the optimal number of clusters from elbow graph...
                opt_cluster = int(st.number_input("Select the optimal cluster value",value=0,key=2))
                
                if opt_cluster:
                    algorithm_pca = (KMeans(n_clusters = opt_cluster ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 100  , algorithm='elkan') )
                    algorithm_pca.fit(principalDf)
                    labels_pca = algorithm_pca.labels_
                    centroids_pca = algorithm_pca.cluster_centers_
                    # Checkbox to view centroid and labels developed from clusters...
                    view_kmdet = st.checkbox("To view the centroid and the labels", value=False)
                    if view_kmdet:
                        st.write(centroids_pca)
                        st.write(np.unique(labels_pca))
                    # Scatter plot for two-dimensional PCA datset...    
                    if len(pcacolname) == 2:
                        fig = px.scatter(principalDf, x=pcacolname[0], y = pcacolname[1], color=labels_pca)
                        st.write(fig)
                    # 3-D Scatter plot for three-dimensional PCA datset...  
                    if len(pcacolname) == 3:
                        fig = px.scatter_3d(principalDf, x = pcacolname[0], y=pcacolname[1], z=pcacolname[2],
                        color=labels_pca, opacity = 0.8, size_max=30)
                        st.write(fig)

    #---------------------------------------------------------------------------------
    # K-Means without Principal Component Analysis...
    if st.sidebar.checkbox("K-Means without PCA",False,key=4):
        st.write("### **• K-Means without Principal Component Analysis**")
        #_______________________________________________________________________________
        # Form creation for users to specify maximum clusters and columns to include...
        with st.form("PCA2_form"):
            collistn = data_scaled.columns.to_list()
            kmcolptions = st.multiselect('Select the columns for K-means', collistn)
            max_clusters = int(st.number_input("Maximum number of Clusters", value=0,key=1))
            submit = st.form_submit_button("Submit")

        new_dataset = dataset[kmcolptions]
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
        opt_cluster = int(st.number_input("Select the optimal cluster value",value=0,key=2))
        if opt_cluster:
            algorithm_pca = (KMeans(n_clusters = opt_cluster ,init='k-means++', n_init = 10 ,max_iter=300, 
                    tol=0.0001,  random_state= 100  , algorithm='elkan') )
            algorithm_pca.fit(new_dataset)
            labels_pca = algorithm_pca.labels_
            centroids_pca = algorithm_pca.cluster_centers_
            # Checkbox to view centroid and labels developed from clusters...
            view_kmdet = st.checkbox("To view the centroid and the labels", value=False)
            if view_kmdet:
                st.write(centroids_pca)
                st.write(np.unique(labels_pca))
            # Scatter plot for two-dimensional PCA datset... 
            if len(kmcolptions) == 2:
                fig = px.scatter(new_dataset, x=kmcolptions[0], y = kmcolptions[1], color=labels_pca)
                st.write(fig)
            # 3-D Scatter plot for three-dimensional PCA datset...  
            if len(kmcolptions) == 3:
                fig = px.scatter_3d(new_dataset, x = kmcolptions[0], y=kmcolptions[1], z=kmcolptions[2],
                color=labels_pca, opacity = 0.8, size_max=30)
                st.write(fig)
    



if __name__ == "__main__":
    main()   




        

    




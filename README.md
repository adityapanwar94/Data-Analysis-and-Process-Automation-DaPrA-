<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- Data-Analysis-and-Process-Automation -->
<br />
<div align="center">
  <a href="https://github.com/adityapanwar94/Data-Analysis-and-Process-Automation-DaPrA-">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Data-Analysis-and-Process-Automation</h3>

  <p align="center">
    <a href="https://github.com/adityapanwar94/Data-Analysis-and-Process-Automation-DaPrA-"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/adityapanwar94/Data-Analysis-and-Process-Automation-DaPrA-">View Demo</a>
    ·
    <a href="https://github.com/adityapanwar94/Data-Analysis-and-Process-Automation-DaPrA-/issues">Report Bug</a>
    ·
    <a href="https://github.com/adityapanwar94/Data-Analysis-and-Process-Automation-DaPrA-/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project



[![Watch the video](https://img.youtube.com/vi/iTu8IUulXRU/hqdefault.jpg)](https://www.youtube.com/watch?v=iTu8IUulXRU)

e-Yantra Robotics Challenge has massive participation every year, with students registering by filling up their user profiles. These profiles gathered information about the users, ranging from their year and branch of engineering to the online courses and the skills they have acquired. This dataset is a source of assigning competition themes to the group of individuals participating as a team. Thus, the project aims at devising automation tools in three stages:
1. Data cleaning and preprocessing pipeline for the user-profile dataset.
2. Data Analysis and clustering dashboard web application for the preprocessed and unlabelled dataset.
3. Machine Learning based model to predict the suitable themes based on the user data.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With
1. Programming Language:
  * [![Python][python.js]][python-url]
2. Technical Tools 
  * [![Streamlit][streamlit.js]][streamlit-url]
  * [![Google Colab][colab.js]][colab-url]
3. Libraries
  * [![Pandas][pandas.js]][pandas-url]
  * [![Numpy][numpy.js]][numpy-url]
  * [![Plotly][plotly.js]][plotly-url]
  * [![Scikit Learn][sklearn.js]][sklearn-url]

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To locally run the various sections of the project follow these simple steps.

### Prerequisites


* Download the packages listed under .streamlit/requirements.txt 
  ```sh
  pip install *name of the package from the list*
  ```

### Installation

1. For running the dashboard locally:
  - Clone the repo
   ```sh
   git clone https://github.com/adityapanwar94/Data-Analysis-and-Process-Automation-DaPrA-
   ```
   - Navigate to the Data_Analysis_Dashboard folder through the terminal
   - Once inside the folder run the following command in the terminal
   ```sh
   streamlit run App.py
   ``` 
2. To implement the prediction model refer the code under  SSL_ModelDevelopment.ipynb notebook
3. Refer to the colab notebooks for data preprocessing pipelines

<p> 
Note: The Data_Analysis_Dashboard folder contains the code for the web application and the associated dataset. Implemention of a new dataset with the dashboard should be according to the following standards:

1. Replacement of null values 
2. Conversion of categorical values to numerics 
</p>

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

1. For dashboard usage and implementation:
_please refer to the [Video](https://www.youtube.com/watch?v=iTu8IUulXRU)_

2. For working around with the model follow the steps given below:
  - Run the SSL_ModelDevelopment.ipynb notebook
  
  - Select the system mode, here a recommendation system generates theme recommendations for preference filling whereas the allotment system specifies the final theme for a team based on their personal data and theme preferences.
  
  ![image](https://user-images.githubusercontent.com/48209998/183883779-c83c80a1-3043-4192-9d96-a57c37628cc3.png)
  - View the results under Meta Estimator Model (Most efficient Model)
  
  ![image](https://user-images.githubusercontent.com/48209998/183884742-bf63a8a0-4c13-438f-a746-593582148ea2.png)
  
  - Final Prediction dataset will be generated below with the predicted theme allotment
  
  - You can go back, change the system type and run the following lines of code again to view the results.


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Section 1- Data Preprocessing pipelines
  - [ ] Dataset_filtering(User_data_only)
     - [ ] Removing teams less than 4 members
     - [ ] Dropping columns with almost all null values
     - [ ] Dropping irrelevant columns
     - [ ] Converting categorical values to numerical values
     - [ ] Function to generate participation frequency
     - [ ] Conversion of 30 different department categories into three 4 categories (Mechanical, Electrical, Computers, Others) 
     - [ ] Correcting CGPA values greater than 10
  - [ ] Dataset_filtering_and_managing_null_values(User_data_only)
      - [ ] Replacing null values 
      - [ ] One hot encoding of categorical columns
      - [ ] Grouping dataset based on team_id 
      - [ ] Final Grouped dataset
  - [ ] Corrected_Computer_Specification(data preprocessing for computer specification same steps followed)
  - [ ] Merging the grouped team data with the team-wise grouped computer specification data  
- [ ] Section 2- Data Analysis Dashboard
  - [ ] Automation of column dropping mechanism 
  - [ ] Interactive bar graph, correlation graph and parallel category plot features
  - [ ] Normalisation of dataset feature
  - [ ] Parameter Weight Defining Mecahnism
  - [ ] K-Means with Principal Component Analysis feature
  - [ ] Simple K-Means
- [ ] Section 3- Model Development
  - [ ] Model as recommender system (without preferences)
  - [ ] Model as allotment system (with preferences)
  - [ ] 3 different algorithms under SSL implemented

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/adityapanwar94/Data-Analysis-and-Process-Automation-DaPrA-/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/adityapanwar94/Data-Analysis-and-Process-Automation-DaPrA-/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/adityapanwar94/Data-Analysis-and-Process-Automation-DaPrA-/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[streamlit.js]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
[streamlit-url]: https://streamlit.io/
[python.js]: https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue
[python-url]: https://www.python.org/
[colab.js]: https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252
[colab-url]: https://research.google.com/colaboratory/
[pandas.js]: https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[numpy.js]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/
[plotly.js]: https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white
[plotly-url]: https://plotly.com/
[sklearn.js]: https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[sklearn-url]: https://scikit-learn.org/stable/
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

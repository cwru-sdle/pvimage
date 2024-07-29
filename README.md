# pvimage

The `pvimage` package aids in the manipulation of electroluminescence images, which is a common characterization technique for PV modules in both the lab and the field. 
However, the process by which Electroluminescent images of photovoltaic modules are captured leads to variation in module orientation between images. To ensure the images are uniformly oriented and registered for analysis, we created an image processing pipeline, which has been discussed in our previous work [1,2,3,4]. Filtering and thresholding methods are used to initially preprocess the data to remove barrel distortion, reduce noise, and remove unimportant background data. With a noise-reduced image, a convex hull algorithm is used to identify cell areas and mark them as a “1” (white pixel) while every other pixel is assigned a “0” (black pixel). A series of 1-D x-axis and y-axis parallel slices are taken through the binary array to identify the steps up (0 to 1) and steps down (1 to 0) across the slice. These steps correspond to the module edge. A regression model is fit to the points along the module edge, and the intersections of the edge lines identify the corners of the PV module. A perspective transformation is, then, applied to uniformly orient and planarize the module image resulting in the final planar-indexed module image ready for subsequent analysis.

# Refrences 

    J.S. Fada, M.A. Hossain, J.L. Braid, S. Yang, T.J. Peshek, R.H. French, Electroluminescent Image Processing and Cell Degradation Type Classification via Computer Vision and Statistical Learning Methodologies, in: 2017 IEEE 44th Photovoltaic Specialist Conference (PVSC), 2017: pp. 3456–3461. https://doi.org/10.1109/PVSC.2017.8366291.

    A.M. Karimi, J.S. Fada, J. Liu, J.L. Braid, M. Koyutürk, R.H. French, Feature Extraction, Supervised and Unsupervised Machine Learning Classification of PV Cell Electroluminescence Images, in: 2018 IEEE 7th World Conference on Photovoltaic Energy Conversion (WCPEC) (A Joint Conference of 45th IEEE PVSC, 28th PVSEC 34th EU PVSEC), 2018: pp. 0418–0424. https://doi.org/10.1109/PVSC.2018.8547739.

    Ahmad M. Karimi, Justin S. Fada, Mohammad A. Hossain, S. Yang, Timothy J. Peshek, Jennifer L. Braid, Roger H. French, Automated Pipeline for Photovoltaic Module Electroluminescence Image Processing and Degradation Feature Classification, IEEE Journal of Photovoltaics. (2019) 1–12. https://doi.org/10.1109/JPHOTOV.2019.2920732.

    Ahmad Maroof Karimi, Justin S. Fada, Nicholas A. Parrilla, Benjamin G. Pierce, Mehmet Koyutürk, Roger H. French, Jennifer L. Braid, Mechanistic Insights into Photovoltaic Module Performance: Application of Computer Vision and Machine Learning on Electroluminescence Images and Current-Voltage Characterization, IEEE Journal of Photovoltaics. (n.d.). https://doi.org/10.1109/JPHOTOV.2020.2973448.

# Contact and Support
Please open a GitHub issue or contact Ben Pierce (pierce@case.edu) for issues, bug reports, and requests. 


# Acknowledgments

This work was supported by the U.S. Department of Energy’s Office of Energy Efficiency and Renewable Energy (EERE) under Solar Energy Technologies Office (SETO) Agreement no. DE-EE-0008172. The work of Jennifer L. Braid was supported by the U.S. Department of Energy (DOE) Office of Energy Efficiency and Renewable Energy administered by the Oak Ridge Institute for Science and Education (ORISE) for the DOE. ORISE is managed by Oak Ridge Associated Universities (ORAU) under DOE Contract no. DE-SC0014664.


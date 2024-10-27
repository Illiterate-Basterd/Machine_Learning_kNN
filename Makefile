SVM = examples\SVM.py
PCA = examples\PCA.py
kNN = task\kNN.py

run_task:
	py $(kNN)

run_svm:
	py $(SVM)

run_pca:
	py $(PCA)
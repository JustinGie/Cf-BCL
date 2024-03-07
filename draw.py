import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import dataloader
from mpl_chord_diagram import chord_diagram

def ImportantROIs(healthFile, patientFile):
    health_roi_score = np.load(healthFile)
    patient_roi_score = np.load(patientFile)
    
    def recordROI(scoresList, Name='Health'):
        topkidx = np.argsort(scoresList)[-10:][::-1]
        ROIs = pd.read_csv('Explain_data/Node_AAL116.node', sep='\t', header=None)
        for idx in topkidx:
            ROIs.loc[idx,4] = 1
        ROIs.to_csv(f'Explain_data/ROI_Edge/{Name}.node', header=None, index=False, sep='\t')
        print(f"top10({Name}): ", [value+1 for value in topkidx])
        print(f"top10({Name}): ", [ROIs[5][idx] for idx in topkidx])
        print(f"top10({Name}): ", [scoresList[idx] for idx in topkidx])
        print("="*100)

    recordROI(health_roi_score, 'Health')
    recordROI(patient_roi_score, 'Patient')

def ImportantEdges(healthFile, patientFile):
    health_roi_score = np.load(healthFile)
    patient_roi_score = np.load(patientFile)

    def recordEdge(scoresList, Name='Health'):
        values = {'Health':0, 'Patient':1}
        topkidx = np.argsort(scoresList)[-10:][::-1]
        dataset = dataloader.MyOwnDataset('data', 'ADHD_ROI116', 'fmri')
        edgeScore = []

        for data in dataset:
            if data.y == values[Name]:
                for i in range(len(data.edge_index[0])):
                    if data.edge_index[0][i].item() in topkidx:
                        edgeScore.append((data.edge_index[0][i].item(), data.edge_index[1][i].item(), scoresList[data.edge_index[0][i]]+ scoresList[data.edge_index[1][i]]))
                sortedList = sorted(edgeScore, key=lambda x: x[-1], reverse=True)[:50]

                out = np.zeros((116,116))
                for values in sortedList:
                    out[values[0], values[1]] = 1
                    out[values[1], values[0]] = 1
                pd.DataFrame(out).to_csv(f'Explain_data/ROI_Edge/{Name}.edge', header=None, index=False, sep='\t')
                break
    recordEdge(health_roi_score, 'Health')
    recordEdge(patient_roi_score, 'Patient')

def drawEdge(file):
    names = pd.read_table("Explain_data/Node_AAL116.node", sep="\t", header=None)
    names = np.array(names[5]).tolist()
    names.append("None")

    matrix = pd.read_table(f"Explain_data/ROI_Edge/{file}.edge", sep="\t", header=None, names=names)
    del matrix["None"]
    temp_matrix = np.array(matrix)

    dictParameter = {
    'rotate_names':True,
    'fontsize':4,
    'min_chord_width':0.5}
    chord_diagram(temp_matrix, names=names[:-1], pad=1.5, gap=0.02, use_gradient=True, cmap="jet", **dictParameter)
    # plt.show()
    plt.savefig(f"Explain_data/ROI_Edge/{file}.png", dpi=300, bbox_inches='tight')


# ImportantROIs("Explain_data/Health_ROI_Aveg.npy", "Explain_data/Patient_ROI_Aveg.npy")
ImportantEdges("Explain_data/Health_ROI_Aveg.npy", "Explain_data/Patient_ROI_Aveg.npy")
drawEdge('Health')
drawEdge('Patient')


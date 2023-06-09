from collections import OrderedDict
import numpy as np

ncpus = 24
nRepeats = 100
kFold = 5

dList =  [ "Arita2018",  "Carvalho2018", \
                "Hosny2018A", "Hosny2018B", "Hosny2018C", \
                "Ramella2018",  "Lu2019","Sasaki2019", "Toivonen2019", "Keek2020", "Li2020", \
                "Park2020", "Song2020", "Veeraraghavan2020" ]



fselParameters = OrderedDict({
    # these are 'one-of'
    "FeatureSelection": {
        "N": [1,2,4,8,16,32,64],
        "Methods": {
            "LASSO": {"C": [1.0]},
            "Anova": {},
            "Bhattacharyya": {},
        }
    }
})

clfParameters = OrderedDict({
    "Classification": {
        "Methods": {
            "LDA": {},
            "LogisticRegression": {"C": np.logspace(-6, 6, 7, base = 2.0) },
            "NaiveBayes": {}
        }
    }
})


#

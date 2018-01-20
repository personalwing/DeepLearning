import numpy as np


def convertSampleToProbMatr(sampleSeq3DArr):  # changed add one column for '1'
    """
    Convertd the raw data to probability matrix

    PARAMETER
    ---------
    sampleSeq3DArr: 3D numpy array
       X denoted the unknow amino acid.


    probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
    """

    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["D"] = 2
    letterDict["E"] = 3
    letterDict["F"] = 4
    letterDict["G"] = 5
    letterDict["H"] = 6
    letterDict["I"] = 7
    letterDict["K"] = 8
    letterDict["L"] = 9
    letterDict["M"] = 10
    letterDict["N"] = 11
    letterDict["P"] = 12
    letterDict["Q"] = 13
    letterDict["R"] = 14
    letterDict["S"] = 15
    letterDict["T"] = 16
    letterDict["V"] = 17
    letterDict["W"] = 18
    letterDict["Y"] = 19
    letterDict["-"] = 20  ##add -
    AACategoryLen = 21  ##add -

    probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))

    sampleNo = 0
    for sequence in sampleSeq3DArr:

        AANo = 0
        for AA in sequence:

            if not AA in letterDict:
                probMatr[sampleNo][0][AANo] = np.full((1, AACategoryLen), 1.0 / AACategoryLen)

            else:
                index = letterDict[AA]
                probMatr[sampleNo][0][AANo][index] = 1

            AANo += 1
        sampleNo += 1

    return probMatr


seq_list = []


def Cut(Seq, Fix_l, Overlap):
    Sep_str = Seq.strip().replace('\n', '')
    l = len(Sep_str)
    if l > Fix_l:
        seq_list.append(Sep_str[:Fix_l])
        Seq_dg = Sep_str[Fix_l - Overlap:]
        return Cut(Seq_dg, Fix_l, Overlap)
    else:
        seq_list.append(Sep_str + (Fix_l - l) * '-')
        return seq_list


def calculate_length(r_filepath):
    """
    remember!!! file end with \n
    :param r_filepath: read_raw_sequence file
    :return: suitable length
    """
    readpath = open(r_filepath, 'r')
    lines = readpath.readlines()
    m = []
    n = []
    a = []
    for line in lines:
        ll = line.rfind('>')
        a.append(len(line[ll+2:]))
    count = len(a)
    myset = set(a)
    i = 0
    for item in myset:
        m.append(item)
        n.append(100*a.count(item)/count)
        i += 1
    local = n.index(max(n))
    # filter values = 0 default
    if m[local] == 0:
        n[local] = 0
    local = n.index((max(n)))
    return m[local]



def Load_file_toNumpy(length, r_filepath, w_filepath):
    readpath = open(r_filepath, 'r')
    OutIns = open(w_filepath, 'w')
    lines = readpath.readlines()
    sa = [[] for h in range(int(len(lines)))]
    i = 0
    for line in lines:
        ll = line.rfind('>')
        # length is the cut sequence and 1 is the overlap size
        sa[i] = Cut(line[ll + 1:], length, 1)
        i += 1
    for item in sa:
        for item_item in item:
            OutIns.write(item_item + '\n')


def readCut(readpath, a):
    f = open(readpath, 'a')
    for i in range(len(a)):
        np.savetxt(f, a[i][0], fmt="%d", newline='\n')
        f.write("\n")


if __name__ == "__main__":
    positive_data_a_filepath = '.\\data\\new_data_label_sequence.txt'

    # length is the suitable cut length
    length = calculate_length(positive_data_a_filepath)
    print(length)

    """
    read raw sequence and cut it
    """
    Load_file_toNumpy(length, positive_data_a_filepath, '.\\data\\new_data_label_sequence_Cut.txt')

    """
     code sequence
    """
    name = '.\\data\\new_data_label_sequence_Cut.txt'
    sampleSeq3DArr = np.loadtxt(name, dtype=str)
    a = convertSampleToProbMatr(sampleSeq3DArr)
    readpath = '.\\data\\result_new_data_label_sequence_Cut.txt'
    readCut(readpath, a)

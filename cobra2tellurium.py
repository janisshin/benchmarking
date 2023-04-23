import os
import glob
import numpy as np

output_directory = 'models/mass_action/teVer/'

model_files = list(glob.glob('models/mass_action/antimony/*.ant'))

for each in model_files:
    print()
    boundaries = []
    kf_initializations = []
    model_text = ''
    with open(each) as model:
        lines = model.readlines()
        for line in lines:
            if line[:2] == 'EX':
                line_split = line.split()
                if line_split[1] == '->':
                    boundaries.append('B' + line_split[2][1:-1])
                    line_split[3] += '*B' + line_split[2][1:-1]
                    line_split[1] = 'B' + line_split[2][1:-1] + ' ' + line_split[1]
                    line = ' '.join(line_split) + '\n'
                    model_text += line
                    continue
                if line_split[2] == '->':
                    # adding the B species in the degradation reaction
                    bdsp = 'B' + line_split[1][1:]
                    # add this B species to the ext line at the top
                    boundaries.append(bdsp)

                    # add in B species to reaction
                    line_split[3] = bdsp + line_split[3]

                    # add in kf portions of rate laws
                    kf_n = line_split[4].split('*')[1][3:]
                    line_split[4] = line_split[4][:-1] + ' - kf' + str(kf_n) + '*' + bdsp + ')'

                    kf_initializations.append('kf' + str(kf_n))
                    
                    line = ' '.join(line_split) + '\n'
                    model_text += line
                    continue
            else:
                model_text += line
    ext_text = 'ext ' + ', '.join(boundaries) + '\n\n'
    model_text = ext_text + model_text
    # print(boundary_text)

    # adding initializations to the bottom of the file
    values_text = ''
    for item in boundaries:
        values_text += item + ' = ' + str(np.random.uniform(0, 10)) + '\n'

    for item in kf_initializations:
        values_text += item + ' = ' + str(np.random.uniform(0, 10)) + '\n'
    model_text += '\n' + values_text

    if not os.path.exists(output_directory): 
        os.makedirs(output_directory)

    file_name = each[:-4].split('/')[-1].split('\\')[-1]
    with open(output_directory + file_name + '_teVer.ant', 'w') as out_model:
        out_model.write(model_text)

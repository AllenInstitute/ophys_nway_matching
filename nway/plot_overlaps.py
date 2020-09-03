import json
import matplotlib.pyplot as plt
import argschema
import numpy as np
import cv2
import PIL.Image

example_args = {
        'input_file': "/allen/programs/braintv/production/neuralcoding/prod0/specimen_791855403/experiment_container_1018027580/OphysNwayCellMatchingStrategy/container_run_1018042567/1018027580_ophys_cell_matching_input.json",
        'output_file': "/allen/aibs/informatics/danielk/nway_test/translation_estimation/1018027580_ophys_cell_matching_output.json"
        }

class MySchema(argschema.ArgSchema):
    input_file = argschema.fields.InputFile(
            required=True,
            desc="input json to nway")
    output_file = argschema.fields.InputFile(
            required=True,
            desc="output json from nway")


def get_images(path_dict, pw_result):
    fexp = pw_result['fixed_experiment']
    mexp = pw_result['moving_experiment']
    with PIL.Image.open(path_dict[fexp]) as im:
        fix = np.array(im)
    with PIL.Image.open(path_dict[mexp]) as im:
        mov = np.array(im)

    tform = np.array(pw_result['transform']['matrix']).astype('float32')
    
    mov = cv2.warpPerspective(
            mov,
            tform,
            fix.shape[::-1],
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP).astype(np.uint8)

    d = {
            'fixed_id': fexp,
            'moving_id': mexp,
            'fixed_im': fix,
            'moving_im': mov
            }

    return d


class Summarizer(argschema.ArgSchemaParser):
    default_schema = MySchema

    def run(self):
        # get the paths to all the images
        with open(self.args['input_file'], "r") as f:
            j = json.load(f)
        impaths = {i['id']: i['ophys_average_intensity_projection_image']
                   for i in j['experiment_containers']['ophys_experiments']}

        # read in the output
        with open(self.args['output_file'], "r") as f:
            outj = json.load(f)

        # make an overlap plot for each pair
        images = [get_images(impaths, i) for i in outj['pairwise_results']]

        # count the number of fixed ids
        fcounts = {k: 0 for k in impaths}
        for i in images:
            fcounts[i['fixed_id']] += 1
        fcounts = {v: k for k, v in fcounts.items()}
        inds = np.array(list(fcounts.keys()))
        inds = np.sort(inds)[::-1]
        nrows = ncols = len(inds) - 1
        pairs = []
        for nr in range(nrows + 1):
            pairs.append([])
            for nc in range(ncols + 1):
                # find which pair
                for i in images:
                    if (fcounts[inds[nr]] == i['fixed_id']) & \
                            (fcounts[inds[nc]] == i['moving_id']):
                                pairs[-1].append(i)
        for p in pairs:
            print(len(p))
                
        print(inds)

        f, a = plt.subplots(nrows, ncols, num=1, clear=True,
                            sharex=True, sharey=True, figsize=(10, 10))

        for i, pr in enumerate(pairs[:-1]):
            print(len(pr))
            for j, p in enumerate(pr):
                a[i, i + j].imshow(0.5 * p['fixed_im'] + 0.5 * p['moving_im'], cmap='gray')
                #a[i, i + j].imshow(0.5* p['moving_im'], cmap='Blues_r', alpha=0.3)
                #+ p['moving_im'])
                a[i, i + j].set_xlabel(p['moving_id'])
                a[i, i + j].set_ylabel(p['fixed_id'])
                a[i, i + j].set_xticks([])
                a[i, i + j].set_yticks([])

        plt.show()


        #for 


if __name__ == "__main__":
    s = Summarizer(input_data=example_args)
    s.run()

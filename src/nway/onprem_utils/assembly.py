import argschema
from nway.onprem_utils import query_utils
from nway.schemas import (CellROISchema, OnPremExperimentSchema,
                          ExperimentContainerSchema,
                          OnPremGeneratedInputSchema)


class OnPremAssemblyInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.Str(default="INFO")
    experiment_ids = argschema.fields.List(
        argschema.fields.Int,
        required=True,
        cli_as_single_argument=True,
        description=("list of experiment IDs. These will be queried to "
                     "assemble an input_json"))


def experiment_query(dbconn, experiment_id):
    rois = dbconn.query(
        f"""
        SELECT id, x, y, width, height, valid_roi as valid, mask_matrix
        FROM cell_rois
        WHERE ophys_experiment_id={experiment_id}""")
    cresult = [CellROISchema().load(roi) for roi in rois]
    stimulus = dbconn.query(
        f"""
        SELECT os.stimulus_name FROM ophys_sessions AS os
        JOIN ophys_experiments AS oe on oe.ophys_session_id=os.id
        WHERE oe.id={experiment_id}""")[0]['stimulus_name']
    avg_projection = dbconn.query(
        f"""
        SELECT wkf.storage_directory || wkf.filename as path
        FROM ophys_cell_segmentation_runs AS ocsr
        JOIN well_known_files AS wkf on wkf.attachable_id=ocsr.id
        JOIN well_known_file_types AS wkft
        ON wkft.id=wkf.well_known_file_type_id
        WHERE wkft.name='OphysAverageIntensityProjectionImage'
        AND ocsr.current=True
        AND ocsr.ophys_experiment_id={experiment_id}""")[0]['path']
    experiment = OnPremExperimentSchema().load({
        'id': experiment_id,
        'stimulus_name': stimulus,
        'ophys_average_intensity_projection_image': avg_projection,
        'cell_rois': cresult})
    return experiment


class OnPremInputAssembly(argschema.ArgSchemaParser):
    default_schema = OnPremAssemblyInputSchema
    default_output_schema = OnPremGeneratedInputSchema

    def run(self, dbconn):
        self.logger.name = type(self).__name__
        experiments = []
        for eid in self.args['experiment_ids']:
            experiments.append(experiment_query(dbconn, eid))
            self.logger.info(f"loaded {len(experiments[-1]['cell_rois'])} "
                             f"ROIs for experiment {eid}")
        container = ExperimentContainerSchema().load(
                {'ophys_experiments': experiments})
        self.output({'experiment_containers': container}, indent=2)
        self.logger.info(f"Wrote {self.args['output_json']}")


if __name__ == "__main__":
    lims_credentials = query_utils.get_db_credentials()
    lims_connection = query_utils.DbConnection(**lims_credentials)
    ia = OnPremInputAssembly()
    ia.run(lims_connection)

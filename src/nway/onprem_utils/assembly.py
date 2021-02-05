import argschema
import marshmallow
import platform
from typing import Union
from pathlib import Path, PureWindowsPath
from nway.onprem_utils import query_utils
from nway.schemas import (CellROISchema, OnPremExperimentSchema,
                          ExperimentContainerSchema,
                          OnPremGeneratedInputSchema)


class OnPremAssemblyException(Exception):
    pass


class OnPremAssemblyInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.Str(default="INFO")
    experiment_ids = argschema.fields.List(
        argschema.fields.Int,
        required=False,
        cli_as_single_argument=True,
        description=("list of experiment IDs. These will be queried to "
                     "assemble an input_json. Either experiment_ids or "
                     "session_ids must be specified."))
    session_ids = argschema.fields.List(
        argschema.fields.Int,
        required=False,
        cli_as_single_argument=True,
        description=("list of session IDs. These will be queried to "
                     "assemble an input_json. If there is not a 1-to-1 "
                     "relationship between session and experiment (for "
                     "example mesoscope sessions) an exception will be "
                     "raised. Either experiment_ids or session_ids must "
                     "be specified."))

    @marshmallow.post_load
    def experiment_or_session(self, data: dict, **kwargs) -> dict:
        keys = ['experiment_ids', 'session_ids']
        if len([k for k in keys if k in data]) != 1:
            raise OnPremAssemblyException("must specify one and only "
                                          f"one of {keys}")
        return data


def system_friendly_filename(fname: Union[Path, str]) -> str:
    if platform.system() == "Windows":
        return "\\" + str(PureWindowsPath(fname))
    else:
        return str(fname)


def experiment_query(dbconn, experiment_id=None, session_id=None):
    if session_id is not None:
        exp_query = dbconn.query(
                f"""
                SELECT id FROM ophys_experiments
                WHERE ophys_session_id={session_id}""")
        if len(exp_query) != 1:
            raise OnPremAssemblyException(
                    "expected a 1-to-1 map of experiment_id to session_id. "
                    f"For session_id {session_id} {len(exp_query)} "
                    "experiments were found")
        experiment_id = exp_query[0]['id']

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
    avg_projection = system_friendly_filename(avg_projection)
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
        if 'session_ids' in self.args:
            call_key = 'session_id'
            iter_ids = self.args['session_ids']
        else:
            call_key = 'experiment_id'
            iter_ids = self.args['experiment_ids']

        experiments = []
        for iter_id in iter_ids:
            experiments.append(
                    experiment_query(dbconn, **{call_key: iter_id}))
            self.logger.info(f"loaded {len(experiments[-1]['cell_rois'])} "
                             f"ROIs for experiment {experiments[-1]['id']}")
        container = ExperimentContainerSchema().load(
                {'ophys_experiments': experiments})
        self.output({'experiment_containers': container}, indent=2)
        self.logger.info(f"Wrote {self.args['output_json']}")


if __name__ == "__main__":
    lims_credentials = query_utils.get_db_credentials()
    lims_connection = query_utils.DbConnection(**lims_credentials)
    ia = OnPremInputAssembly()
    ia.run(lims_connection)

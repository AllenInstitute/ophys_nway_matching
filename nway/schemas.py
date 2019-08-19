from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
        Boolean, Int, Str, Float, List,
        Dict, InputFile, OutputDir, Nested)
import marshmallow as mm


class NwayMatchingSchema(ArgSchema):
    ''' Class that uses argschema to take care of input arguments '''

    save_registered_image = Boolean(
        required=False,
        default=True,
        description='Whether to save registered image.')
    save_pairwise_tables = Boolean(
        required=False,
        default=False,
        description=("Whether to save matching tables from "
                     "pairwise matching"))
    maximum_distance = Int(
        required=False,
        default=10,
        description=("Maximum distance (in pixels) between two cells,"
                     " above which a match is always rejected."))
    diagnostic_figures = Boolean(
        required=False,
        default=False,
        desciption='Plot diagnostic figures.')
    registration_iterations = Int(
        required=False,
        default=1000,
        description=("Number of iterations for intensity-"
                     "based registration"))
    registration_precision = Float(
        required=False,
        default=1.5e-7,
        description=("Threshold of squared error, below which "
                     "registration is terminated"))
    munkres_executable = Str(
        required=False,
        missing=None,
        default=None,
        description=("Executable of Kuhn-Munkres algorithm for bipartite"
                     "graph matching with path information"))
    id_pattern = Str(
        required=False,
        missing='ophys_experiment_\d+',
        default='ophys_experiment_\d+',
        description=("passed to re.findall() as search pattern "
                     "on the input intensity filename and used "
                     "for naming intermediate and output files. "))
    motionType = Str(
        required=False,
        missing="MOTION_AFFINE",
        default="MOTION_AFFINE",
        validator=mm.validate.OneOf([
            "MOTION_TRANSLATION",
            "MOTION_EUCLIDEAN",
            "MOTION_AFFINE",
            "MOTION_HOMOGRAPHY"
            ]),
        description=("motion model passed to cv2.findTransformECC"
                     "during image registration step."))

class ExperimentSchema(DefaultSchema):
    id = Int(
        required=True,
        description="experiment id")
    ophys_average_intensity_projection_image = InputFile(
        required=True,
        description="max projection intensity image")
    max_int_mask_image = InputFile(
        required=True,
        description="mask image")
    cell_rois = Dict(
        required=True,
        description='dict mapping of ids, labels, zs,')


class PairwiseMatchingSchema(NwayMatchingSchema):
    output_directory = OutputDir(
        required=True,
        description="destination for output files")
    fixed = Nested(ExperimentSchema)
    moving = Nested(ExperimentSchema)

from argschema import ArgSchema
from argschema.fields import Boolean, Int, Str, Float


class NwayMatchingSchema(ArgSchema):
    ''' Class that uses argschema to take care of input arguments '''

    save_registered_image = Boolean(
        required=False,
        default=True,
        description='Whether to save registered image.')
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

import numpy as np
from scipy.special import binom

time_interval = 1/25

# !!! use 3D velocity
def calculate_velocity(df):
    velocities = []
    for i in range(1, len(df)):
        distance = np.sqrt((df.iloc[i]['E_Distance'] - df.iloc[i-1]['E_Distance'])**2 +
                        (df.iloc[i]['F_Offset'] - df.iloc[i-1]['F_Offset'])**2 +
                        (df.iloc[i]['G_Height'] - df.iloc[i-1]['G_Height'])**2)
        velocity = distance / time_interval
        velocities.append(velocity)
        # add NaN for the first velocity as it has no previous point
    return velocities

def find_end(df):
    """
    Find the end point of the ball path.
    """
    velocity = calculate_velocity(df)
    peak_velocity_index = velocity.index(max(velocity))
    end_index = peak_velocity_index + 2
    return end_index 

def bernstein_poly(i, n, t):
    """
    Compute the Bernstein polynomial B_i^n at t
    """
    return binom(n, i) * ( t**i ) * ( (1 - t)**(n - i) )
    
def bezier_curve(control_points, num_points=100):
    """
    Compute the Bezier curve for given control points.
    - control_points: a list of coordinates that define the shape of the bezier curve
    - num_points: # of points to compute along the curve (smoothness)
    """
    n = len(control_points) - 1 # degree of the curve 
    t = np.linspace(0, 1, num_points)
    xvals = np.zeros_like(t)
    yvals = np.zeros_like(t)
    for i, (x, y) in enumerate(control_points):
        bernstein = bernstein_poly(i, n, t)
        xvals += bernstein * x
        yvals += bernstein * y
    return xvals, yvals

def bezier_derivative(control_points, num_points=100):
    """
    Compute the first and second derivatives of the Bezier curve.
    """
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points) # parameterize the curve with t
    dx = np.zeros_like(t)
    dy = np.zeros_like(t)
    ddx = np.zeros_like(t)
    ddy = np.zeros_like(t)
    # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
    # for i, (x, y) in enumerate(control_points):
    #     dx += n * (bernstein_poly(i-1, n-1, t) - bernstein_poly(i, n-1, t)) * x
    #     dy += n * (bernstein_poly(i-1, n-1, t) - bernstein_poly(i, n-1, t)) * y
    #     ddx += n * (n-1) * (bernstein_poly(i-1, n-2, t) - 2 * bernstein_poly(i, n-2, t) + bernstein_poly(i+1, n-2, t)) * x
    #     ddy += n * (n-1) * (bernstein_poly(i-1, n-2, t) - 2 * bernstein_poly(i, n-2, t) + bernstein_poly(i+1, n-2, t)) * y
    
    # https://en.wikipedia.org/wiki/B%C3%A9zier_curve#:~:text=B%C3%A9zier%20curves%20are%20widely%20used,to%20manipulate%20the%20curve%20intuitively.
    for i in range(n):
        P_diff_x = control_points[i+1][0] - control_points[i][0]
        P_diff_y = control_points[i+1][1] - control_points[i][1]
        dx += n * P_diff_x * bernstein_poly(i, n-1, t)
        dy += n * P_diff_y * bernstein_poly(i, n-1, t)

    for i in range(n-1):
        P_diff_x = control_points[i+2][0] - 2 * control_points[i+1][0] + control_points[i][0]
        P_diff_y = control_points[i+2][1] - 2 * control_points[i+1][1] + control_points[i][1]
        ddx += n * (n-1) * P_diff_x * bernstein_poly(i, n-2, t)
        ddy += n * (n-1) * P_diff_y * bernstein_poly(i, n-2, t)
    
    return dx, dy, ddx, ddy

def bezier_curvature(control_points, t):
    """
    Calculate the curvature of the Bezier curve at a given t.
    """
    num_points = 100
    dx, dy, ddx, ddy = bezier_derivative(control_points, num_points)
    
    x_t = np.interp(t, np.linspace(0, 1, num_points), dx)
    y_t = np.interp(t, np.linspace(0, 1, num_points), dy)
    x_tt = np.interp(t, np.linspace(0, 1, num_points), ddx)
    y_tt = np.interp(t, np.linspace(0, 1, num_points), ddy)
    
    # https://en.wikipedia.org/wiki/Curvature
    curvature = np.abs(x_t * y_tt - y_t * x_tt) / (x_t ** 2 + y_t ** 2) ** 1.5
    if np.isnan(curvature) or np.isinf(curvature):
        curvature = 0
    
    return curvature
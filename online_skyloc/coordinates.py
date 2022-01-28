from __future__ import division
import numpy as np

def cartesian_to_spherical(vector):
    """Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """
    r = np.linalg.norm(vector, axis = -1)
    unit = np.array([v/np.linalg.norm(v) for v in vector])
    theta = np.arccos(unit[:,2])
    phi = np.arctan2(unit[:,1], unit[:,0])
    coord = np.array([r,theta,phi]).T
    return coord


def spherical_to_cartesian(vector):
    """Convert the spherical coordinate vector [r, theta, phi] to the Cartesian vector [x, y, z].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param spherical_vect:  The spherical coordinate vector [r, theta, phi].
    @type spherical_vect:   3D array or list
    @param cart_vect:       The Cartesian vector [x, y, z].
    @type cart_vect:        3D array or list
    """
    # Trig alias.
    sin_theta = np.sin(vector[:,1])
    # The vector.
    x = vector[:,2] * np.cos(vector[:,0]) * sin_theta
    y = vector[:,2] * np.sin(vector[:,0]) * sin_theta
    z = vector[:,2] * np.cos(vector[:,1])
    return np.array([x,y,z]).T
    
def celestial_to_cartesian(celestial_vect):
    """Convert the spherical coordinate vector [r, dec, ra] to the Cartesian vector [x, y, z]."""
    celestial_vect = np.atleast_2d(celestial_vect)
    celestial_vect[:,1] = np.pi/2. - celestial_vect[:,1]
    return spherical_to_cartesian(celestial_vect)

def cartesian_to_celestial(cartesian_vect):
    """Convert the Cartesian vector [x, y, z] to the celestial coordinate vector [r, dec, ra]."""
    cartesian_vect = np.atleast_2d(cartesian_vect)
    spherical_vect = cartesian_to_spherical(cartesian_vect)
    spherical_vect[:,1] = np.pi/2. - spherical_vect[:,1]
    D   = spherical_vect[:,0]
    dec = spherical_vect[:,1]
    ra  = spherical_vect[:,2]
    return np.array([ra, dec, D]).T

def Jacobian(cartesian_vect):
    cartesian_vect = np.atleast_2d(cartesian_vect)
    d = np.linalg.norm(cartesian_vect, axis = -1)
    d_sin_theta = np.linalg.norm(cartesian_vect[:,:-1])
    return d*d_sin_theta

def inv_Jacobian(celestial_vect):
    celestial_vect = np.atleast_2d(celestial_vect)
    cartesian_vect = celestial_to_cartesian(celestial_vect)
    detJ = Jacobian(cartesian_vect)
    return 1/detJ

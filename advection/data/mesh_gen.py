from dolfin import *
from mshr import *

# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
subdomain1 = Rectangle(Point(0.0, 0.0), Point(1.0, 0.25))
subdomain2 = Rectangle(Point(0.0, 0.25), Point(1.0, 1.0))
domain.set_subdomain(1, subdomain1)  # add some fake subdomains to make sure that the mesh is split
domain.set_subdomain(2, subdomain2)  # at x[1] = 0.25, since boundary id changes at (0, 0.25)

# governs how fine mesh is (larger=finer mesh)
mesh_density = 75

mesh = generate_mesh(domain, mesh_density)

print("Mesh generated")
print(f"Num vertices: {mesh.num_vertices()}")
print(f"Num elements: {mesh.num_cells()}")
print(f"Max mesh element size: {mesh.hmax()}")
print(f"Min mesh element size: {mesh.hmin()}")

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)

# Create boundaries
class Boundary1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            abs(x[1] - 0.) < DOLFIN_EPS
            or (abs(x[0] - 0.) < DOLFIN_EPS and x[1] <= 0.25)
        )


class Boundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            abs(x[1] - 1.) < DOLFIN_EPS
            or abs(x[0] - 1.) < DOLFIN_EPS
            or (abs(x[0] - 0.) < DOLFIN_EPS and x[1] >= 0.25)
        )


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
boundary_1 = Boundary1()
boundary_1.mark(boundaries, 1)
boundary_2 = Boundary2()
boundary_2.mark(boundaries, 2)

File(f"./mesh.xml") << mesh
File(f"./subdomains.xml") << subdomains
File(f"./boundaries.xml") << boundaries
# XDMFFile(f"./data/mesh.xdmf").write(mesh)
# XDMFFile(f"./data/subdomains.xdmf").write(subdomains)
# XDMFFile(f"./data/boundaries.xdmf").write(boundaries)
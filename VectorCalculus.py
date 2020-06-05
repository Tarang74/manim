from manimlib.imports import *

DISTANCE_COLOR = BLUE
TIME_COLOR = YELLOW
VELOCITY_COLOR = GREEN

class FathersOfVectorCalculus(Scene):
    CONFIG = {
        "names": [
            "Gibbs",
            "Heaviside",
        ],
        "picture_height": 3,
    }

    def construct(self):
        title = TextMobject("Vector Calculus")
        title.set_height(0.8)
        title.to_edge(UP)
        self.add(title)

        men = Mobject()
        for name in self.names:
            image = ImageMobject(name, invert=False)
            image.set_height(self.picture_height)
            title = TextMobject(name)
            title.scale(0.8)
            title.next_to(image, DOWN)
            image.add(title)
            men.add(image)
        men.arrange(RIGHT, aligned_edge=UP)
        men.shift(DOWN)

        discover_brace = Brace(men[:2], UP)
        discover = discover_brace.get_text("Discoverers")
        VGroup(discover_brace, discover).set_color(YELLOW)
        rigor_brace = Brace(men[0], DOWN)
        rigor = rigor_brace.get_text("Created the notation")
        rigor.shift(0.1 * DOWN)
        VGroup(rigor_brace, rigor).set_color(YELLOW)

        for man in men:
            self.play(FadeIn(man))
        self.play(GrowFromCenter(discover_brace), Write(discover, run_time=1))
        self.play(GrowFromCenter(rigor_brace), Write(rigor, run_time=1))
        self.wait()


def Range(in_val,end_val,step=1):
    return list(np.arange(in_val,end_val+step,step))

class CalculusReview(GraphScene):
    CONFIG = {
        "y_max" : 8,
        "y_min" : -8,
        "x_max" : 4,
        "x_min" : -4,
        "y_tick_frequency" : 2,
        "x_tick_frequency" : 1,
        "graph_origin" : ORIGIN,
        "y_axis_label": None,
        "x_axis_label": None,
        "x_axis_width": 13,
        "x_coords": []
    }
    def construct(self):
        self.setup_axes()
        self.plotFunc = self.get_graph(lambda x : 2*x**5+2*x**4-5.5*x**3-4*x**2+1.5*x+2.5, 
                                    color = GREEN,
                                    x_min=-2.2,
                                    x_max=1.8,
                                )
        label = self.get_graph_label(self.plotFunc)
        self.plotFunc.set_stroke(width=3) # width of line

        # Animation
        for plot in (self.plotFunc):
            self.play(
                    ShowCreation(plot),
                    run_time = 2
                )
        self.play(Write(label))
        self.wait()
        self.remove(label)
        
        self.show_tangent_line()
    
    def show_tangent_line(self):
        def f(x):
            return 2*x**5+2*x**4-5.5*x**3-4*x**2+1.5*x+2.5
        
        def g(x):
            return 10*x**4+8*x**3-16.5*x**2-8*x+1.5

        dots = list(map(Dot, points))
        self.play(ShowCreation(dots[0]), ShowCreation(dots[1]))
        xval = 0.8
        
        line = self.get_graph(lambda x: -3*(x - xval) + f(xval), x_min=-4, x_max=4)
        self.play(ShowCreation(line), run_time=1)
        
        self.ind = 0
        def approx_tangent(line, dt):
            alpha = interpolate(-3, g(0.8), dt)
            line_new = self.get_graph(lambda x: alpha*(x - xval)+f(xval), color=BLUE, x_min=-3, x_max=3)
            line.become(line_new)

        self.play(UpdateFromAlphaFunc(line, approx_tangent), run_time=3)

        def move_tangent(line, dt):
            alpha = interpolate(xval,-0.8, dt)
            line_new = self.get_graph(lambda x: g(alpha)*(x - alpha) + f(alpha), color=BLUE, x_min=-3, x_max=3)
            line.become(line_new)

        self.play(UpdateFromAlphaFunc(line, move_tangent), run_time=5, rate_func=there_and_back)

    def setup_axes(self):
        GraphScene.setup_axes(self)
        # width of edges
        self.x_axis.set_stroke(width=2)
        self.y_axis.set_stroke(width=2)
        # color of edges
        self.x_axis.set_color(BLUE)
        self.y_axis.set_color(BLUE)
        # Add x,y labels
        ax = TexMobject("x")
        ax.set_color(WHITE)
        ax.next_to(self.x_axis,RIGHT)
        ay = TexMobject("y")
        ay.set_color(WHITE)
        ay.next_to(self.y_axis,UP)


        # Y labels
        self.y_axis.label_direction = LEFT*1.5
        self.y_axis.add_numbers(*[-8, -4, 4, 8])
        self.x_axis.label_direction = DOWN*1.5
        self.x_axis.add_numbers(*[-4,-2,2,4])
        origin = TexMobject("0")
        origin.set_height(0.25)
        origin.set_color(WHITE)
        origin.move_to(0.5*LEFT+0.5*DOWN)
        
        
        self.play(
            *[Write(objectO)
            for objectO in [
                    self.y_axis,
                    self.x_axis,
                    ax, ay, origin
                ]
            ],
            run_time=2
        )


class VectorReview(Scene):
    def construct(self):
        vector, symbol, coordinates = self.intro_vector()
    
    def intro_vector(self):
        VI = 1
        VJ = 2

        plane = NumberPlane()
        labels = VMobject(*plane.get_coordinate_labels())
        vector = Vector(VI * RIGHT + VJ * UP, color=RED)
        coordinates = vector_coordinate_labels(vector)
        symbol = TexMobject("\\vec{\\textbf{v}}")
        symbol.move_to(0.8 * (RIGHT + UP))

        self.play(ShowCreation(plane, lag_ratio=1, run_time=3))
        self.play(ShowCreation(vector))
        self.play(Write(labels), Write(coordinates), Write(symbol))
        self.wait(2)
        return vector, symbol, coordinates

class VectorFieldScene1(Scene):
    def construct(self):
        func = lambda p: np.array([
            p[0]/2,  # x
            p[1]/2,  # y
            0        # z
        ])
        # Normalized
        vector_field_norm = VectorField(func)
        # Not normalized
        vector_field_not_norm = VectorField(func, length_func=linear)
        self.play(*[GrowArrow(vec) for vec in vector_field_norm])
        self.wait(2)
        self.play(ReplacementTransform(vector_field_norm,vector_field_not_norm))
        self.wait(2)

# Other way
def functioncurlreal(p, velocity=0.05):
    x, y = p[:2]
    result =  - y * RIGHT + x * UP
    result *= velocity
    return result

class VectorFieldScene2(Scene):
    def construct(self):
        vector_field = VectorField(functioncurlreal)
        dot1 = Dot([1,1,0], color=RED)
        dot2 = Dot([2,2,0], color=BLUE)
        self.add(vector_field,dot1,dot2)
        self.wait()
        for dot in dot1,dot2:
            move_submobjects_along_vector_field(
                dot,
                lambda p: functioncurlreal(p,0.5)
            )
        self.wait(3)
        for dot in dot1,dot2:
            dot.clear_updaters()
        self.wait()

class VectorFieldScene3(Scene):
    def construct(self):
        vector_field = VectorField(
            lambda p: functioncurlreal(p,0.5),
            length_func = lambda norm: 0.6 * sigmoid(norm)
        )
        dot = Dot([2,3,0]).fade(1)
        some_vector = vector_field.get_vector(dot.get_center())
        some_vector.add_updater(
            lambda mob: mob.become(vector_field.get_vector(dot.get_center()))
        )
        self.add(vector_field,dot,some_vector)
        self.play(
            dot.shift,LEFT*4,
            run_time=3
        )
        self.play(
            dot.shift,DOWN*5,
            run_time=3
        )
        self.play(
            Rotating(dot, radians=PI, about_point=ORIGIN),
            run_time=5
        )
        self.wait()


def get_force_field_func(*point_strength_pairs, **kwargs):
    radius = kwargs.get("radius", 0.5)

    def func(point):
        result = np.array(ORIGIN)
        for center, strength in point_strength_pairs:
            to_center = center - point
            norm = get_norm(to_center)
            if norm == 0:
                continue
            elif norm < radius:
                to_center /= radius**3
            elif norm >= radius:
                to_center /= norm**3
            to_center *= -strength
            result += to_center
        return result
    return func

class ElectricParticle(Circle):
    CONFIG = {
        "color": WHITE,
        "sign": "+",
    }
    def __init__(self, radius=0.5 ,**kwargs):
        digest_config(self, kwargs)
        super().__init__(
            stroke_color=WHITE,
            stroke_width=0.5,
            fill_color=self.color,
            fill_opacity=0.8,
            radius=radius
        )
        sign = TexMobject(self.sign)
        sign.set_stroke(WHITE, 1)
        sign.set_width(0.5 * self.get_width())
        sign.move_to(self)
        self.add(sign)

class Proton(ElectricParticle):
    CONFIG = {
        "color": RED_E,
    }

class Electron(ElectricParticle):
    CONFIG = {
        "color": BLUE_E,
        "sign": "-"
    }

class ChangingElectricField(Scene):
    CONFIG = {
        "vector_field_config": {},
        "num_particles": 6,
        "anim_time": 5,
    }
    def construct(self):
        particles = self.get_particles()
        vector_field = self.get_vector_field()

        def update_vector_field(vector_field):
            new_field = self.get_vector_field()
            vector_field.become(new_field)
            vector_field.func = new_field.func

        # The dt parameter will be explained in 
        # future videos, but here is a small preview.
        def update_particles(particles, dt):
            func = vector_field.func
            for particle in particles:
                force = func(particle.get_center())
                particle.velocity += force * dt
                particle.shift(particle.velocity * dt)

        vector_field.add_updater(update_vector_field),
        particles.add_updater(update_particles),
        self.add(
            vector_field,
            particles
        )
        # Animation time:
        self.wait(self.anim_time)
        # Suspend animation
        for mob in vector_field,particles:
            mob.suspend_updating()
        self.wait()
        # Restore animation
        for mob in vector_field,particles:
            mob.resume_updating()
        self.wait(3)


    def get_particles(self):
        particles = self.particles = VGroup()
        for n in range(self.num_particles):
            if n % 2 == 0:
                particle = Proton(radius=0.2)
                particle.charge = +1
            else:
                particle = Electron(radius=0.2)
                particle.charge = -1
            particle.velocity = np.random.normal(0, 0.1, 3)
            particles.add(particle)
            particle.shift(np.random.normal(0, 0.2, 3))

        particles.arrange_in_grid(buff=LARGE_BUFF)
        return particles

    def get_vector_field(self):
        func = get_force_field_func(*list(zip(
            list(map(lambda x: x.get_center(), self.particles)),
            [p.charge for p in self.particles]
        )))
        self.vector_field = VectorField(func, **self.vector_field_config)
        return self.vector_field

class FunctionTrackerWithNumberLine(Scene):
    def construct(self):
        # f(x) = x**2
        fx = lambda x: x.get_value()**2
        # ValueTrackers definition
        x_value = ValueTracker(0)
        fx_value = ValueTracker(fx(x_value))
        # DecimalNumber definition
        x_tex = DecimalNumber(x_value.get_value()).add_updater(lambda v: v.set_value(x_value.get_value()))
        fx_tex = DecimalNumber(fx_value.get_value()).add_updater(lambda v: v.set_value(fx(x_value)))
        # TeX labels definition
        x_label = TexMobject("x = ")
        fx_label = TexMobject("x^2 = ")
        # Grouping of labels and numbers
        group = VGroup(x_tex,fx_tex,x_label,fx_label).scale(2)
        # Set the labels position
        x_label.next_to(x_tex,LEFT, buff=0.7,aligned_edge=x_label.get_bottom())
        fx_label.next_to(fx_tex,LEFT, buff=0.7,aligned_edge=fx_label.get_bottom())
        # Grouping numbers and labels
        x_group = VGroup(x_label,x_tex)
        fx_group = VGroup(fx_label,fx_tex)
        # Align labels and numbers
        VGroup(x_group, fx_group).arrange_submobjects(RIGHT,buff=2,aligned_edge=DOWN).to_edge(UP)
        # Get NumberLine,Arrow and label from x
        x_number_line_group = self.get_number_line_group(
            "x",30,0.2,step_label=10,v_tracker=x_value,tick_frequency=2
            )
        x_number_line_group.to_edge(LEFT,buff=1)
        # Get NumberLine,Arrow and label from f(x)
        fx_number_line_group = self.get_number_line_group(
            "x^2",900,0.012,step_label=100,v_tracker=fx_tex,
            tick_frequency=50
            )
        fx_number_line_group.next_to(x_number_line_group,DOWN,buff=1).to_edge(LEFT,buff=1)

        self.add(
            x_number_line_group,
            fx_number_line_group,
            group
            )
        self.wait()
        self.play(
            x_value.set_value,30,
            rate_func=linear,
            run_time=10
            )
        self.wait()
        self.play(
            x_value.set_value,0,
            rate_func=linear,
            run_time=10
            )
        self.wait(3)


    def get_number_labels_to_numberline(self,number_line,x_max=None,x_min=0,buff=0.2,step_label=1,**tex_kwargs):
        # This method return the labels of the NumberLine
        labels = VGroup()
        x_max = number_line.x_max
        for x in range(x_min,x_max+1,step_label):
            x_label = TexMobject(f"{x}",**tex_kwargs)
            # See manimlib/mobject/number_line.py CONFIG dictionary
            x_label.next_to(number_line.number_to_point(x),DOWN,buff=buff)
            labels.add(x_label)
        return labels

    def get_number_line_group(self,label,x_max,unit_size,v_tracker,step_label=1,**number_line_config):
        # Set the Label (x,or x**2)
        number_label = TexMobject(label)
        # Set the arrow 
        arrow = Arrow(UP,DOWN,buff=0).set_height(0.5)
        # Set the number_line
        number_line = NumberLine(
            x_min=0,
            x_max=x_max,
            unit_size=unit_size,
            numbers_with_elongated_ticks=[],
            **number_line_config
            )
        # Get the labels from number_line
        labels = self.get_number_labels_to_numberline(number_line,step_label=step_label,height=0.2)
        # Set the arrow position
        arrow.next_to(number_line.number_to_point(0),UP,buff=0)
        # Grouping arrow and number_label
        label = VGroup(arrow,number_label)
        # Set the position of number_label
        number_label.next_to(arrow,UP,buff=0.1)
        # Grouping all elements
        number_group = VGroup(label,number_line,labels)
        # Set the updater to the arrow and number_label
        label.add_updater(lambda mob: mob.next_to(number_line.number_to_point(v_tracker.get_value()),UP,buff=0))

        return number_group

class ThreeDScene(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        circle=Circle()
        self.set_camera_orientation(phi=80 * DEGREES, theta=45 * DEGREES)
        self.play(ShowCreation(axes), ShowCreation(circle))

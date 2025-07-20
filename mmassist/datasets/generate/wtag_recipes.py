MUG_CAKE = """Mug Cake

Ingredients
2 Tablespoons all-purpose flour
1.5 Tablespoons granulated sugar
1/4 teaspoon baking powder
Pinch salt
2 teaspoons canola or vegetable oil
2 Tablespoons water
1/4 teaspoon vanilla extract
Container of chocolate frosting (premade)

Tools and Utensils
measuring spoons small mixing bowl whisk
paper cupcake liner 12-ounce coffee mug plate
microwave
zip-top bag, snack or sandwich size scissors
spoon
toothpick

Steps
1. Place the paper cupcake liner inside the mug. Set aside.
2. Measure and add the flour, sugar, baking powder, and salt to the mixing bowl.
3. Whisk to combine.
4. Measure and add the oil, water, and vanilla to the bowl.
5. Whisk batter until no lumps remain.
6. Pour batter into prepared mug.
7. Microwave the mug and batter on high power for 60 seconds.
8. Check if the cake is done by inserting and toothpick into the center of the cake and then
removing. If wet batter clings to the toothpick, microwave for an additional 5 seconds. If the
toothpick comes out clean, continue.
9. Invert the mug to release the cake onto a plate. Allow to cool until it is no longer hot to the
touch, then carefully remove paper liner.
10. While the cake is cooling, prepare to pipe the frosting. Scoop 4 spoonfuls of chocolate frosting
into a zip-top bag and seal, removing as much air as possible.
11. Use scissors to cut one corner from the bag to create a small opening 1/4-inch in diameter.
12. Squeeze the frosting through the opening to apply small dollops of frosting to the plate in a
circle around the base of the cake."""

COFFEE = """Pour-over Coffee

Ingredients
12 oz water
25 grams whole coffee beans

Tools and Utensils
2-cup liquid measuring cup electric kettle
kitchen scale
coffee grinder
filter cone dripper (stainless steel)
paper basket filter (standard 8-12 cup size) 12-ounce coffee mug
thermometer
timer (optional)

Steps
1. Measure 12 ounces of cold water and transfer to a kettle.
2. While the water is boiling, assemble the filter cone. Place the dripper on top of a coffee mug.
3. Prepare the filter insert by folding the paper filter in half to create a semi-circle, and in half again
to create a quarter-circle. Place the paper filter in the dripper and spread open to create a cone.
4. Weigh the coffee beans and grind until the coffee grounds are the consistency of coarse sand,
about 20 seconds. Transfer the grounds to the filter cone.
5. Once the water has boiled, check the temperature. The water should be between 195-205
degrees Fahrenheit or between 91-96 degrees Celsius. If the water is too hot, let it cool briefly.
6. Pour a small amount of water in the filter to wet the grounds. Wait about 30 seconds for coffee
to bloom. You will see small bubbles or foam on the coffee grounds during this step.
7. Slowly pour the rest of the water over the grounds in a circular motion. Do not overfill beyond
the top of the paper filter.
8. Let the coffee drain completely into the mug before removing the dripper. Discard the paper
filter and coffee grounds."""


PINWHEELS = """Pinwheels

Ingredients
1 8-inch flour tortilla
Jar of nut butter or allergy-friendly alternative (such as sunbutter, soy butter, or seed butter) Jar of jelly, jam, or fruit preserves

Tools and Utensils
cutting board
butter knife
paper towel
toothpicks
~12-inch strand of dental floss plate

Steps
1. Place tortilla on cutting board.
2. Use a butter knife to scoop nut butter from the jar. Spread nut butter onto tortilla, leaving 1/2-
inch uncovered at the edges.
3. Clean the knife by wiping with a paper towel.
4. Use the knife to scoop jelly from the jar. Spread jelly over the nut butter.
5. Clean the knife by wiping with a paper towel.
6. Roll the tortilla from one end to the other into a log shape, about 1.5 inches thick. Roll it tight
enough to prevent gaps, but not so tight that the filling leaks.
7. Secure the rolled tortilla by inserting 5 toothpicks about 1 inch apart.
8. Trim the ends of the tortilla roll with the butter knife, leaving 1⁄2 inch margin between the last
toothpick and the end of the roll. Discard ends.
9. Slide floss under the tortilla, perpendicular to the length of the roll. Place the floss halfway
between two toothpicks.
10. Cross the two ends of the floss over the top of the tortilla roll. Holding one end of the floss in
each hand, pull the floss ends in opposite directions to slice.
11. Continue slicing with floss to create 5 pinwheels.
12. Place the pinwheels on a plate."""


def get_task_and_recipe(step_descriptions: str) -> tuple[str, str]:
    step_descriptions = step_descriptions.lower()
    if "coffee" in step_descriptions:
        return "Make pour-over coffee", COFFEE
    elif "tortilla" in step_descriptions:
        return "Make pinwheels", PINWHEELS
    return "Make mug cake", MUG_CAKE

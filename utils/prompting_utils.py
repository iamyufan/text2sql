import random
import torch


def create_few_shot_prompt(text, schema_description, train_texts, train_queries, k):
    prompt = f"Database: {schema_description}\n"
    prompt += "Using valid SQL, answer the following questions for the tables provided above. Please do not generate more information other than the SQL queries.\n"
    sample_indices = random.sample(range(len(train_texts)), k)
    examples = [
        f"Query: {train_texts[i].strip()}\nSQL: ```sql{train_queries[i].strip()}\n```\n"
        for i in sample_indices
    ]
    examples = "\n".join(examples)
    prompt += examples
    prompt += f"Query: {text.strip()}\nSQL: ```sql\n"
    return prompt


# Function to generate SQL queries
def generate_sql_queries(
    loader,
    schema_description,
    model,
    tokenizer,
    train_texts,
    train_queries,
    k=1,
    max_new_tokens=256,
    test=False,
):
    model.eval()  # Set model to evaluation mode
    generated_queries = []

    counter = 0
    total_batches = len(loader)
    with torch.no_grad():
        for texts, queries in loader:
            batch_prompts = [
                create_few_shot_prompt(
                    text, schema_description, train_texts, train_queries, k
                )
                for text in texts
            ]
            batch_input_ids = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).input_ids.to(model.device)
            batch_outputs = model.generate(
                batch_input_ids, max_new_tokens=max_new_tokens
            )
            batch_sql_queries = [
                tokenizer.decode(output, skip_special_tokens=True)
                for output in batch_outputs
            ]
            generated_queries.extend(batch_sql_queries)

            counter += 1
            if counter % 10 == 0:
                print(f"Processed {counter}/{total_batches} batches")

    return generated_queries


# Function to extract the SQL query from the generated text
def extract_sql_query(generated_text):
    # Split the text into sections based on ```sql and ```
    sections = generated_text.split("```sql\n")
    # The last SQL section before the closing ```
    last_section = sections[-1].split("\n```")[0]

    last_section = last_section.strip()
    last_section = last_section.replace("\n", " ")
    return last_section.strip()


schema_description = """
Table name: restriction
  - Columns: no_discounts, minimum_stay, stopovers, restriction_code, application, maximum_stay, saturday_stay_required, advance_purchase

Table name: flight_stop
  - Columns: departure_airline, stop_number, arrival_flight_number, flight_id, arrival_time, departure_flight_number, stop_time, arrival_airline, stop_days, stop_airport, departure_time

Table name: food_service
  - Columns: meal_code, compartment, meal_number, meal_description

Table name: month
  - Columns: month_number, month_name

Table name: code_description
  - Columns: code, description

Table name: city
  - Columns: city_name, country_name, state_code, time_zone_code, city_code

Table name: flight_fare
  - Columns: flight_id, fare_id, restriction_code, to_airport, from_airport, one_direction_cost, round_trip_cost, fare_basis_code, fare_airline

Table name: state
  - Columns: country_name, state_code, state_name

Table name: fare_basis
  - Columns: discounted, class_type, season, basis_days, booking_class, night, premium, fare_basis_code, economy

Table name: date_day
  - Columns: day_number, month_number, day_name, year

Table name: time_interval
  - Columns: period, end_time, begin_time

Table name: flight
  - Columns: to_airport, aircraft_code_sequence, dual_carrier, flight_id, stops, flight_days, connections, arrival_time, time_elapsed, flight_number, from_airport, airline_flight, airline_code, meal_code, departure_time

Table name: dual_carrier
  - Columns: low_flight_number, high_flight_number, main_airline, service_name, dual_airline

Table name: aircraft
  - Columns: aircraft_code, capacity, wing_span, engines, aircraft_description, basic_type, weight, pressurized, length, propulsion, pay_load, cruising_speed, range_miles, wide_body, manufacturer

Table name: fare
  - Columns: to_airport, restriction_code, round_trip_required, fare_id, from_airport, one_direction_cost, round_trip_cost, fare_basis_code, fare_airline

Table name: compartment_class
  - Columns: compartment, class_type

Table name: flight_leg
  - Columns: flight_id, leg_number, leg_flight

Table name: days
  - Columns: days_code, day_name

Table name: airport_service
  - Columns: minutes_distant, airport_code, direction, city_code, miles_distant

Table name: airport
  - Columns: airport_code, airport_name, airport_location, minimum_connect_time, country_name, state_code, time_zone_code

Table name: time_zone
  - Columns: time_zone_name, hours_from_gmt, time_zone_code

Table name: airline
  - Columns: note, airline_code, airline_name

Table name: equipment_sequence
  - Columns: aircraft_code, aircraft_code_sequence

Table name: ground_service
  - Columns: airport_code, transport_type, city_code, ground_fare

Table name: class_of_service
  - Columns: booking_class, class_description, rank
"""

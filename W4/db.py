import psycopg2

#connect to the db
con = psycopg2.connect(
            host = "localhost",
            database="alphaLibertee",
            user = "postgres",
            password = "yourrealpassword")

#cursor
cur = con.cursor()

# cur.execute("insert into users (id, name, income) values (%s, %s, %s)", (2, "Mike", 1000) )

#execute query
cur.execute("select id, name, income from users")

rows = cur.fetchall()

for r in rows:
    print (f"id {r[0]} name {r[1]} income {r[2]}")

#commit the transcation
con.commit()

#close the cursor
cur.close()

#close the connection
con.close()